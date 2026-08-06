#include "gs_renderer.h"

#include "core/logger.hpp"
#include "viewport_scratch_bucket.h"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace {
    namespace indirect = lfs::rendering::vulkan::indirect_layout;

    constexpr size_t kRasterBatchSize = RASTER_BATCH_SIZE;
    constexpr size_t kRasterDenseTileThreshold = RASTER_DENSE_TILE_THRESHOLD;
    constexpr size_t kMinLoadBalancedRasterInstances = 4 * kRasterBatchSize;
    constexpr size_t kMinLoadBalancedAverageTileInstances = kRasterBatchSize / 16;
    constexpr uint32_t kInstanceCountOverflowSentinel = std::numeric_limits<uint32_t>::max();

    class ConditionalRenderingScope {
    public:
        ConditionalRenderingScope(VulkanGSPipeline& pipeline,
                                  const bool enabled,
                                  const PFN_vkCmdBeginConditionalRenderingEXT begin,
                                  const PFN_vkCmdEndConditionalRenderingEXT end,
                                  const _VulkanBuffer& predicate_buffer,
                                  const VkDeviceSize predicate_offset)
            : pipeline_(&pipeline),
              command_buffer_(pipeline.activeCommandBuffer()),
              end_(end),
              active_(enabled) {
            if (!active_)
                return;

            const VkConditionalRenderingBeginInfoEXT begin_info{
                .sType = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT,
                .pNext = nullptr,
                .buffer = predicate_buffer.buffer,
                .offset = predicate_buffer.offset + predicate_offset,
                .flags = 0,
            };
            begin(command_buffer_, &begin_info);
        }

        ConditionalRenderingScope(const ConditionalRenderingScope&) = delete;
        ConditionalRenderingScope& operator=(const ConditionalRenderingScope&) = delete;

        ~ConditionalRenderingScope() noexcept {
            // If a nested guard already cancelled this recording, reset discarded
            // the whole command buffer and there is no live scope to close. On
            // every normal or propagating path, close this wave before the owning
            // DeviceGuard can submit or cancel the batch.
            if (active_ && pipeline_->isCommandBatchInProgress() &&
                pipeline_->activeCommandBuffer() == command_buffer_) {
                end_(command_buffer_);
            }
        }

    private:
        VulkanGSPipeline* pipeline_;
        VkCommandBuffer command_buffer_;
        PFN_vkCmdEndConditionalRenderingEXT end_;
        bool active_;
    };

    [[nodiscard]] size_t denseTileBatchCapacity(const size_t tile_instances,
                                                const size_t num_tiles) {
        const size_t max_dense_tiles =
            std::min(num_tiles, tile_instances / (kRasterDenseTileThreshold + 1u));
        return std::max<size_t>(1, _CEIL_DIV(tile_instances, kRasterBatchSize) + max_dense_tiles);
    }

    [[nodiscard]] _VulkanBuffer bufferView(const _VulkanBuffer& buffer,
                                           const VkDeviceSize relative_offset,
                                           const VkDeviceSize size) {
        if (!buffer.containsRange(relative_offset, size)) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "VkSplat buffer view is outside its source (buffer={:#x}, base_offset={}, relative_offset={}, view_bytes={}, source_bytes={}, allocation_bytes={}, label='{}')",
                    lfs::rendering::vkHandleValue(buffer.buffer),
                    buffer.offset,
                    relative_offset,
                    size,
                    buffer.size,
                    buffer.allocSize,
                    buffer.label ? buffer.label : "<unlabeled>"),
                LFS_SOURCE_SITE_CURRENT());
        }
        _VulkanBuffer view = buffer;
        view.offset += relative_offset;
        view.capacity = static_cast<size_t>(size);
        view.size = static_cast<size_t>(size);
        return view;
    }

    void validateIndirectLayoutBuffer(const _VulkanBuffer& buffer,
                                      const indirect::Layout layout,
                                      const std::string_view operation) {
        const std::size_t required_bytes = indirect::byteSize(layout);
        if (buffer.size < required_bytes || !buffer.containsRange(0, required_bytes)) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "{} requires a live indirect-buffer view satisfying {} (buffer={:#x}, layout_constant='{}', required_words={}, required_bytes={}, active_bytes={}, view_capacity={}, backing_bytes={}, base_offset={}, label='{}')",
                    operation,
                    layout.word_count_constant,
                    lfs::rendering::vkHandleValue(buffer.buffer),
                    layout.word_count_constant,
                    layout.word_count,
                    required_bytes,
                    buffer.size,
                    buffer.capacity,
                    buffer.allocSize,
                    buffer.offset,
                    buffer.label ? buffer.label : "<unlabeled>"),
                LFS_SOURCE_SITE_CURRENT());
        }
    }
} // namespace

VulkanGSRenderer::VulkanGSRenderer()
    : VulkanGSPipeline() {
}

VulkanGSRenderer::~VulkanGSRenderer() noexcept {
    cancelCommandBatch();
    try {
        waitForPendingBatch();
        cleanup();
    } catch (const std::exception& error) {
        fprintf(stderr, "VulkanGSRenderer cleanup failed: %s\n", error.what());
    } catch (...) {
        fprintf(stderr, "VulkanGSRenderer cleanup failed with an unknown error\n");
    }
}

void VulkanGSRenderer::cleanup() {
    destroyVisibleCountReadback();
    destroyLodSelectionReadback();
    destroyInstanceCountReadback();
    destroyInstanceGateReadback();
    VulkanGSPipeline::cleanup();
}

void VulkanGSRenderer::tagDeferredVisibleCountReadback(const VkSemaphore semaphore,
                                                       const std::uint64_t value) {
    // Tag only the frame whose command buffer contains the copy. Re-tagging
    // every frame ratchets the awaited timeline value past GPU completion
    // whenever rendering is continuous, and the stats starve.
    if (visible_count_readback_pending_ &&
        visible_count_readback_signal_ == VK_NULL_HANDLE) {
        if (semaphore == VK_NULL_HANDLE || value == 0) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "Visible-count readback requires a valid completion timeline tag (semaphore={:#x}, value={}, pending={}, prior_semaphore={:#x}, prior_value={})",
                    lfs::rendering::vkHandleValue(semaphore),
                    value,
                    visible_count_readback_pending_,
                    lfs::rendering::vkHandleValue(visible_count_readback_signal_),
                    visible_count_readback_value_),
                LFS_SOURCE_SITE_CURRENT());
        }
        visible_count_readback_signal_ = semaphore;
        visible_count_readback_value_ = value;
    }
}

void VulkanGSRenderer::tagDeferredLodSelectionReadback(const VkSemaphore semaphore,
                                                       const std::uint64_t value) {
    if (lod_selection_readback_pending_) {
        if (semaphore == VK_NULL_HANDLE || value == 0) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "LOD-selection readback requires a valid completion timeline tag (semaphore={:#x}, value={}, pending={}, prior_semaphore={:#x}, prior_value={})",
                    lfs::rendering::vkHandleValue(semaphore),
                    value,
                    lod_selection_readback_pending_,
                    lfs::rendering::vkHandleValue(lod_selection_readback_signal_),
                    lod_selection_readback_value_),
                LFS_SOURCE_SITE_CURRENT());
        }
        lod_selection_readback_signal_ = semaphore;
        lod_selection_readback_value_ = value;
    }
}

void VulkanGSRenderer::tagDeferredInstanceCountReadback(const VkSemaphore semaphore,
                                                        const std::uint64_t value) {
    if (instance_count_readback_pending_ &&
        instance_count_readback_signal_ == VK_NULL_HANDLE) {
        if (semaphore == VK_NULL_HANDLE || value == 0) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "Tile-instance readback requires a valid completion timeline tag (semaphore={:#x}, value={}, pending={}, prior_semaphore={:#x}, prior_value={})",
                    lfs::rendering::vkHandleValue(semaphore),
                    value,
                    instance_count_readback_pending_,
                    lfs::rendering::vkHandleValue(instance_count_readback_signal_),
                    instance_count_readback_value_),
                LFS_SOURCE_SITE_CURRENT());
        }
        instance_count_readback_signal_ = semaphore;
        instance_count_readback_value_ = value;
    }
}

void VulkanGSRenderer::ensureInstanceCountReadback() {
    if (instance_count_readback_initialized_)
        return;

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = 3 * sizeof(uint32_t);
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo alloc_info{};
    instance_count_readback_buffer_.label = "instance_count_readback";
    const VkResult create_result = vmaCreateBuffer(allocator, &info, &aci,
                                                   &instance_count_readback_buffer_.buffer,
                                                   &instance_count_readback_buffer_.allocation,
                                                   &alloc_info);
    if (create_result != VK_SUCCESS) {
        instance_count_readback_buffer_.buffer = VK_NULL_HANDLE;
        instance_count_readback_buffer_.allocation = VK_NULL_HANDLE;
        lfs::rendering::throw_vk_result(
            create_result,
            "vmaCreateBuffer",
            std::format(
                "Tile-instance readback buffer allocation failed (requested_bytes={}, allocator={:#x}, result={}({}))",
                info.size,
                lfs::rendering::vkHandleValue(allocator),
                lfs::rendering::vkResultToString(create_result),
                static_cast<int>(create_result)),
            LFS_SOURCE_SITE_CURRENT());
    }
    instance_count_readback_buffer_.allocSize = info.size;
    instance_count_readback_buffer_.capacity = info.size;
    instance_count_readback_buffer_.size = info.size;
    instance_count_readback_mapped_ = static_cast<uint32_t*>(alloc_info.pMappedData);
    if (instance_count_readback_mapped_ == nullptr) {
        const VkBuffer failed_buffer = instance_count_readback_buffer_.buffer;
        const VmaAllocation failed_allocation = instance_count_readback_buffer_.allocation;
        vmaDestroyBuffer(allocator,
                         failed_buffer,
                         failed_allocation);
        instance_count_readback_buffer_ = {};
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Tile-instance readback allocation was not persistently mapped (requested_bytes={}, buffer={:#x}, allocation={:#x}, mapped_pointer={:#x})",
                info.size,
                lfs::rendering::vkHandleValue(failed_buffer),
                lfs::rendering::vkHandleValue(failed_allocation),
                lfs::rendering::vkHandleValue(instance_count_readback_mapped_)),
            LFS_SOURCE_SITE_CURRENT());
    }
    instance_count_readback_mapped_[0] = 0;
    instance_count_readback_mapped_[1] = 0;
    instance_count_readback_mapped_[2] = 0;
    setDebugObjectName(VK_OBJECT_TYPE_BUFFER,
                       instance_count_readback_buffer_.buffer,
                       "vksplat.readback.tile_instance_count");
    instance_count_readback_initialized_ = true;
    instance_count_readback_pending_ = false;
    instance_count_readback_signal_ = VK_NULL_HANDLE;
    instance_count_readback_value_ = 0;
}

void VulkanGSRenderer::destroyInstanceCountReadback() {
    if (!instance_count_readback_initialized_)
        return;
    if (instance_count_readback_buffer_.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator,
                         instance_count_readback_buffer_.buffer,
                         instance_count_readback_buffer_.allocation);
    }
    instance_count_readback_buffer_ = {};
    instance_count_readback_mapped_ = nullptr;
    instance_count_readback_initialized_ = false;
    instance_count_readback_pending_ = false;
    instance_count_readback_signal_ = VK_NULL_HANDLE;
    instance_count_readback_value_ = 0;
}

void VulkanGSRenderer::recordInstanceCountReadback(VulkanGSPipelineBuffers& buffers,
                                                   const size_t armed) {
    ensureInstanceCountReadback();
    const auto& count_buffer = buffers.tile_sort_count.deviceBuffer;
    if (count_buffer.buffer == VK_NULL_HANDLE ||
        count_buffer.size != sizeof(uint32_t)) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Tile-instance readback requires the raw-count word (buffer={:#x}, offset={}, active_bytes={}, allocation_bytes={}, required_bytes={})",
                lfs::rendering::vkHandleValue(count_buffer.buffer),
                count_buffer.offset,
                count_buffer.size,
                count_buffer.allocSize,
                sizeof(uint32_t)),
            LFS_SOURCE_SITE_CURRENT());
    }
    // Never stomp an in-flight tagged copy: with the GPU a frame behind, the
    // re-record would reset the tag every frame and the stats would starve.
    if (instance_count_readback_pending_ &&
        instance_count_readback_signal_ != VK_NULL_HANDLE)
        return;

    using lfs::rendering::vulkan::BufferUse;
    using lfs::rendering::vulkan::DeclaredAccess;

    const auto& wave_buffer = buffers.depth_wave_dispatch.deviceBuffer;
    const DeclaredAccess pre_copy[] = {
        {.buffer = &count_buffer, .use = BufferUse::TransferRead},
        {.buffer = &wave_buffer, .use = BufferUse::TransferRead},
        {.buffer = &instance_count_readback_buffer_, .use = BufferUse::TransferWrite},
    };
    planTransfer(std::span{pre_copy});

    VkBufferCopy copy{};
    copy.srcOffset = buffers.tile_sort_count.deviceBuffer.offset;
    copy.dstOffset = 0;
    copy.size = sizeof(uint32_t);
    validateBufferRange(count_buffer, 0, copy.size, "tile-instance count readback source");
    validateBufferRange(instance_count_readback_buffer_, 0, copy.size, "tile-instance count readback destination");
    if (vulkan_dispatch_.cmd_copy_buffer == nullptr) {
        lfs::rendering::throw_renderer_contract(
            "recordInstanceCountReadback requires VulkanDispatch::cmd_copy_buffer",
            LFS_SOURCE_SITE_CURRENT());
    }
    vulkan_dispatch_.cmd_copy_buffer(command_buffer,
                                     buffers.tile_sort_count.deviceBuffer.buffer,
                                     instance_count_readback_buffer_.buffer,
                                     1,
                                     &copy);
    const VkDeviceSize needed_offset =
        indirect::byteOffset(indirect::DepthWave::kHeaderNeededWord);
    validateBufferRange(wave_buffer,
                        needed_offset,
                        sizeof(uint32_t),
                        "depth-wave needed-count readback source");
    VkBufferCopy wave_copy{};
    wave_copy.srcOffset = wave_buffer.offset + needed_offset;
    wave_copy.dstOffset = sizeof(uint32_t);
    wave_copy.size = sizeof(uint32_t);
    vulkan_dispatch_.cmd_copy_buffer(command_buffer,
                                     wave_buffer.buffer,
                                     instance_count_readback_buffer_.buffer,
                                     1,
                                     &wave_copy);
    const uint32_t armed_u32 = static_cast<uint32_t>(armed);
    validateBufferRange(instance_count_readback_buffer_,
                        2 * sizeof(uint32_t),
                        sizeof(uint32_t),
                        "depth-wave armed-count readback destination");
    // Host-side word write; plan TransferWrite already covers the dst buffer.
    vkCmdUpdateBuffer(command_buffer,
                      instance_count_readback_buffer_.buffer,
                      2 * sizeof(uint32_t),
                      sizeof(uint32_t),
                      &armed_u32);
    const DeclaredAccess host_read{
        .buffer = &instance_count_readback_buffer_,
        .use = BufferUse::HostRead,
    };
    planTransfer(std::span{&host_read, 1});
    instance_count_readback_pending_ = true;
    instance_count_readback_signal_ = VK_NULL_HANDLE;
    instance_count_readback_value_ = 0;
}

std::optional<VulkanGSRenderer::TileInstanceStats>
VulkanGSRenderer::pollDeferredTileInstanceStats() {
    if (!instance_count_readback_pending_ || !instance_count_readback_mapped_)
        return std::nullopt;
    if (instance_count_readback_signal_ == VK_NULL_HANDLE || instance_count_readback_value_ == 0)
        return std::nullopt;
    if (!timelineValueComplete(instance_count_readback_signal_, instance_count_readback_value_))
        return std::nullopt;
    if (!invalidateReadbackBuffer(instance_count_readback_buffer_, 3 * sizeof(uint32_t)))
        return std::nullopt;

    TileInstanceStats stats{};
    stats.count_overflow = instance_count_readback_mapped_[0] == kInstanceCountOverflowSentinel;
    stats.raw_count = stats.count_overflow ? 0u : instance_count_readback_mapped_[0];
    stats.waves_needed = instance_count_readback_mapped_[1];
    stats.waves_armed = instance_count_readback_mapped_[2];
    instance_count_readback_pending_ = false;
    instance_count_readback_signal_ = VK_NULL_HANDLE;
    instance_count_readback_value_ = 0;
    return stats;
}

void VulkanGSRenderer::ensureInstanceGateReadback() {
    if (instance_gate_readback_initialized_)
        return;

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = sizeof(uint32_t);
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo alloc_info{};
    instance_gate_readback_buffer_.label = "instance_gate_readback";
    const VkResult create_result = vmaCreateBuffer(allocator,
                                                   &info,
                                                   &aci,
                                                   &instance_gate_readback_buffer_.buffer,
                                                   &instance_gate_readback_buffer_.allocation,
                                                   &alloc_info);
    if (create_result != VK_SUCCESS) {
        instance_gate_readback_buffer_ = {};
        lfs::rendering::throw_vk_result(
            create_result,
            "vmaCreateBuffer",
            std::format(
                "Export tile-instance gate allocation failed (requested_bytes={}, allocator={:#x}, result={}({}))",
                info.size,
                lfs::rendering::vkHandleValue(allocator),
                lfs::rendering::vkResultToString(create_result),
                static_cast<int>(create_result)),
            LFS_SOURCE_SITE_CURRENT());
    }
    instance_gate_readback_buffer_.allocSize = info.size;
    instance_gate_readback_buffer_.capacity = info.size;
    instance_gate_readback_buffer_.size = info.size;
    instance_gate_readback_mapped_ = static_cast<uint32_t*>(alloc_info.pMappedData);
    if (instance_gate_readback_mapped_ == nullptr) {
        vmaDestroyBuffer(allocator,
                         instance_gate_readback_buffer_.buffer,
                         instance_gate_readback_buffer_.allocation);
        instance_gate_readback_buffer_ = {};
        lfs::rendering::throw_renderer_contract(
            "Export tile-instance gate allocation was not persistently mapped",
            LFS_SOURCE_SITE_CURRENT());
    }
    instance_gate_readback_mapped_[0] = 0;
    setDebugObjectName(VK_OBJECT_TYPE_BUFFER,
                       instance_gate_readback_buffer_.buffer,
                       "vksplat.readback.export_tile_instance_gate");
    instance_gate_readback_initialized_ = true;
}

void VulkanGSRenderer::destroyInstanceGateReadback() {
    if (!instance_gate_readback_initialized_)
        return;
    if (instance_gate_readback_buffer_.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator,
                         instance_gate_readback_buffer_.buffer,
                         instance_gate_readback_buffer_.allocation);
    }
    instance_gate_readback_buffer_ = {};
    instance_gate_readback_mapped_ = nullptr;
    instance_gate_readback_initialized_ = false;
}

VulkanGSRenderer::TileInstanceGate VulkanGSRenderer::synchronizeTileInstanceGate(
    VulkanGSPipelineBuffers& buffers) {
    if (!commandBatchInProgress) {
        lfs::rendering::throw_renderer_contract(
            "Export tile-instance gate requires active batch A",
            LFS_SOURCE_SITE_CURRENT());
    }
    ensureInstanceGateReadback();
    const auto& count = buffers.tile_sort_count.deviceBuffer;
    if (count.buffer == VK_NULL_HANDLE || count.size != sizeof(uint32_t)) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Export tile-instance gate requires the raw-count word (buffer={:#x}, bytes={})",
                lfs::rendering::vkHandleValue(count.buffer),
                count.size),
            LFS_SOURCE_SITE_CURRENT());
    }

    using lfs::rendering::vulkan::BufferUse;
    using lfs::rendering::vulkan::DeclaredAccess;

    const DeclaredAccess pre_copy[] = {
        {.buffer = &count, .use = BufferUse::TransferRead},
        {.buffer = &instance_gate_readback_buffer_, .use = BufferUse::TransferWrite},
    };
    planTransfer(std::span{pre_copy});

    const VkBufferCopy copy{
        .srcOffset = count.offset,
        .dstOffset = 0,
        .size = sizeof(uint32_t),
    };
    validateBufferRange(count, 0, copy.size, "export tile-instance gate source");
    validateBufferRange(instance_gate_readback_buffer_,
                        0,
                        copy.size,
                        "export tile-instance gate destination");
    if (vulkan_dispatch_.cmd_copy_buffer == nullptr) {
        lfs::rendering::throw_renderer_contract(
            "synchronizeTileInstanceGate requires VulkanDispatch::cmd_copy_buffer",
            LFS_SOURCE_SITE_CURRENT());
    }
    vulkan_dispatch_.cmd_copy_buffer(command_buffer,
                                     count.buffer,
                                     instance_gate_readback_buffer_.buffer,
                                     1,
                                     &copy);
    const DeclaredAccess host_read{
        .buffer = &instance_gate_readback_buffer_,
        .use = BufferUse::HostRead,
    };
    planTransfer(std::span{&host_read, 1});

    // This is the single intentional export stall: batch A has produced the
    // exact raw count. The dedicated gate has no deferred never-stomp state.
    endCommandBatch(/*use_fence=*/true);
    waitForPendingBatch();
    if (!invalidateReadbackBuffer(instance_gate_readback_buffer_, copy.size)) {
        lfs::rendering::throw_renderer_contract(
            "Export tile-instance gate mapped allocation invalidation failed",
            LFS_SOURCE_SITE_CURRENT());
    }

    TileInstanceGate gate{};
    gate.count_overflow =
        instance_gate_readback_mapped_[0] == kInstanceCountOverflowSentinel;
    gate.raw_count = gate.count_overflow ? 0u : instance_gate_readback_mapped_[0];
    beginCommandBatch();
    return gate;
}

void VulkanGSRenderer::ensureVisibleCountReadback() {
    if (visible_count_readback_initialized_)
        return;

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = 2 * sizeof(uint32_t);
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo alloc_info{};
    visible_count_readback_buffer_.label = "visible_count_readback";
    const VkResult create_result = vmaCreateBuffer(allocator, &info, &aci,
                                                   &visible_count_readback_buffer_.buffer,
                                                   &visible_count_readback_buffer_.allocation,
                                                   &alloc_info);
    if (create_result != VK_SUCCESS) {
        visible_count_readback_buffer_.buffer = VK_NULL_HANDLE;
        visible_count_readback_buffer_.allocation = VK_NULL_HANDLE;
        lfs::rendering::throw_vk_result(
            create_result,
            "vmaCreateBuffer",
            std::format(
                "Visible-count readback buffer allocation failed (requested_bytes={}, allocator={:#x}, result={}({}))",
                info.size,
                lfs::rendering::vkHandleValue(allocator),
                lfs::rendering::vkResultToString(create_result),
                static_cast<int>(create_result)),
            LFS_SOURCE_SITE_CURRENT());
    }
    visible_count_readback_buffer_.allocSize = 2 * sizeof(uint32_t);
    visible_count_readback_buffer_.capacity = 2 * sizeof(uint32_t);
    visible_count_readback_buffer_.size = 2 * sizeof(uint32_t);
    visible_count_readback_mapped_ = static_cast<uint32_t*>(alloc_info.pMappedData);
    if (visible_count_readback_mapped_ == nullptr) {
        const VkBuffer failed_buffer = visible_count_readback_buffer_.buffer;
        const VmaAllocation failed_allocation = visible_count_readback_buffer_.allocation;
        vmaDestroyBuffer(allocator, failed_buffer, failed_allocation);
        visible_count_readback_buffer_ = {};
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Visible-count readback allocation was not persistently mapped (requested_bytes={}, buffer={:#x}, allocation={:#x}, mapped_pointer={:#x})",
                info.size,
                lfs::rendering::vkHandleValue(failed_buffer),
                lfs::rendering::vkHandleValue(failed_allocation),
                lfs::rendering::vkHandleValue(visible_count_readback_mapped_)),
            LFS_SOURCE_SITE_CURRENT());
    }
    visible_count_readback_mapped_[0] = 0;
    visible_count_readback_mapped_[1] = 0;
    setDebugObjectName(VK_OBJECT_TYPE_BUFFER,
                       visible_count_readback_buffer_.buffer,
                       "vksplat.readback.visible_count");
    visible_count_readback_initialized_ = true;
    visible_count_readback_pending_ = false;
    visible_count_readback_signal_ = VK_NULL_HANDLE;
    visible_count_readback_value_ = 0;
    visible_count_readback_num_splats_ = 0;
}

void VulkanGSRenderer::destroyVisibleCountReadback() {
    if (!visible_count_readback_initialized_)
        return;
    if (visible_count_readback_buffer_.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator,
                         visible_count_readback_buffer_.buffer,
                         visible_count_readback_buffer_.allocation);
    }
    visible_count_readback_buffer_ = {};
    visible_count_readback_mapped_ = nullptr;
    visible_count_readback_initialized_ = false;
    visible_count_readback_pending_ = false;
    visible_count_readback_signal_ = VK_NULL_HANDLE;
    visible_count_readback_value_ = 0;
    visible_count_readback_num_splats_ = 0;
}

void VulkanGSRenderer::ensureLodSelectionReadback(const size_t chunk_capacity) {
    if (lod_selection_readback_initialized_) {
        if (lod_selection_readback_chunk_capacity_ >= chunk_capacity)
            return;
        // Growing requires a recreate; never destroy under an in-flight copy.
        if (lod_selection_readback_pending_)
            return;
        destroyLodSelectionReadback();
    }

    const VkDeviceSize byte_size = (2 + chunk_capacity) * sizeof(uint32_t);
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = byte_size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo alloc_info{};
    lod_selection_readback_buffer_.label = "lod_selection_readback";
    const VkResult create_result = vmaCreateBuffer(allocator, &info, &aci,
                                                   &lod_selection_readback_buffer_.buffer,
                                                   &lod_selection_readback_buffer_.allocation,
                                                   &alloc_info);
    if (create_result != VK_SUCCESS) {
        lod_selection_readback_buffer_.buffer = VK_NULL_HANDLE;
        lod_selection_readback_buffer_.allocation = VK_NULL_HANDLE;
        lfs::rendering::throw_vk_result(
            create_result,
            "vmaCreateBuffer",
            std::format(
                "LOD-selection readback buffer allocation failed (requested_bytes={}, payload_words={}, allocator={:#x}, result={}({}))",
                byte_size,
                2 + chunk_capacity,
                lfs::rendering::vkHandleValue(allocator),
                lfs::rendering::vkResultToString(create_result),
                static_cast<int>(create_result)),
            LFS_SOURCE_SITE_CURRENT());
    }
    lod_selection_readback_buffer_.allocSize = byte_size;
    lod_selection_readback_buffer_.capacity = byte_size;
    lod_selection_readback_buffer_.size = byte_size;
    lod_selection_readback_mapped_ = static_cast<uint32_t*>(alloc_info.pMappedData);
    if (lod_selection_readback_mapped_ == nullptr) {
        const VkBuffer failed_buffer = lod_selection_readback_buffer_.buffer;
        const VmaAllocation failed_allocation = lod_selection_readback_buffer_.allocation;
        vmaDestroyBuffer(allocator, failed_buffer, failed_allocation);
        lod_selection_readback_buffer_ = {};
        lfs::rendering::throw_renderer_contract(
            std::format(
                "LOD-selection readback allocation was not persistently mapped (requested_bytes={}, buffer={:#x}, allocation={:#x}, mapped_pointer={:#x})",
                byte_size,
                lfs::rendering::vkHandleValue(failed_buffer),
                lfs::rendering::vkHandleValue(failed_allocation),
                lfs::rendering::vkHandleValue(lod_selection_readback_mapped_)),
            LFS_SOURCE_SITE_CURRENT());
    }
    std::memset(lod_selection_readback_mapped_, 0, byte_size);
    setDebugObjectName(VK_OBJECT_TYPE_BUFFER,
                       lod_selection_readback_buffer_.buffer,
                       "vksplat.readback.lod_selection");
    lod_selection_readback_initialized_ = true;
    lod_selection_readback_pending_ = false;
    lod_selection_readback_signal_ = VK_NULL_HANDLE;
    lod_selection_readback_value_ = 0;
    lod_selection_readback_capacity_ = 0;
    lod_selection_readback_chunk_capacity_ = chunk_capacity;
}

void VulkanGSRenderer::destroyLodSelectionReadback() {
    if (!lod_selection_readback_initialized_)
        return;
    if (lod_selection_readback_buffer_.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator,
                         lod_selection_readback_buffer_.buffer,
                         lod_selection_readback_buffer_.allocation);
    }
    lod_selection_readback_buffer_ = {};
    lod_selection_readback_mapped_ = nullptr;
    lod_selection_readback_initialized_ = false;
    lod_selection_readback_pending_ = false;
    lod_selection_readback_signal_ = VK_NULL_HANDLE;
    lod_selection_readback_value_ = 0;
    lod_selection_readback_capacity_ = 0;
    lod_selection_readback_chunk_capacity_ = 0;
}

std::optional<VulkanGSRenderer::PrimitiveVisibilityStats>
VulkanGSRenderer::pollDeferredPrimitiveVisibilityStats() {
    // Consume only after the tagged render-completion timeline has signaled;
    // otherwise keep the previous stats and avoid a CPU-side GPU drain.
    if (!visible_count_readback_pending_ || !visible_count_readback_mapped_)
        return std::nullopt;
    if (visible_count_readback_signal_ == VK_NULL_HANDLE || visible_count_readback_value_ == 0)
        return std::nullopt;
    if (!timelineValueComplete(visible_count_readback_signal_, visible_count_readback_value_))
        return std::nullopt;
    if (!invalidateReadbackBuffer(visible_count_readback_buffer_, 2 * sizeof(uint32_t)))
        return std::nullopt;

    PrimitiveVisibilityStats stats{};
    stats.num_splats = visible_count_readback_num_splats_;
    stats.visible_count = std::min<size_t>(visible_count_readback_mapped_[0], stats.num_splats);
    stats.raw_count = std::min<size_t>(visible_count_readback_mapped_[1], stats.num_splats);
    visible_count_readback_pending_ = false;
    visible_count_readback_signal_ = VK_NULL_HANDLE;
    visible_count_readback_value_ = 0;
    return stats;
}

std::optional<VulkanGSRenderer::LodSelectionStats>
VulkanGSRenderer::pollDeferredLodSelectionStats() {
    if (!lod_selection_readback_pending_ || !lod_selection_readback_mapped_)
        return std::nullopt;
    if (lod_selection_readback_signal_ == VK_NULL_HANDLE || lod_selection_readback_value_ == 0)
        return std::nullopt;
    if (!timelineValueComplete(lod_selection_readback_signal_, lod_selection_readback_value_))
        return std::nullopt;
    if (!invalidateReadbackBuffer(
            lod_selection_readback_buffer_,
            (2 + 4 + kLodCompactProtectedCap + 2 * kLodCompactMissCap) * sizeof(uint32_t)))
        return std::nullopt;

    LodSelectionStats stats{};
    stats.candidate_count = lod_selection_readback_mapped_[0];
    stats.rendered_capacity = lod_selection_readback_capacity_;
    stats.overflow_count = lod_selection_readback_mapped_[1];
    const uint32_t* const words = lod_selection_readback_mapped_;
    const size_t protected_count =
        std::min<size_t>(words[2], kLodCompactProtectedCap);
    const size_t miss_count = std::min<size_t>(words[3], kLodCompactMissCap);
    stats.protected_overflow = words[4];
    stats.miss_overflow = words[5];
    stats.protected_chunks.assign(words + 6, words + 6 + protected_count);
    stats.miss_candidates.reserve(miss_count);
    const uint32_t* const misses = words + 6 + kLodCompactProtectedCap;
    for (size_t i = 0; i < miss_count; ++i) {
        stats.miss_candidates.emplace_back(misses[i * 2], misses[i * 2 + 1]);
    }
    lod_selection_readback_pending_ = false;
    lod_selection_readback_signal_ = VK_NULL_HANDLE;
    lod_selection_readback_value_ = 0;
    lod_selection_readback_capacity_ = 0;
    return stats;
}

void VulkanGSRenderer::recordVisibleCountReadback(VulkanGSPipelineBuffers& buffers,
                                                  const size_t num_splats) {
    ensureVisibleCountReadback();
    const auto& count_buffer = buffers.visible_count.deviceBuffer;
    if (count_buffer.buffer == VK_NULL_HANDLE || count_buffer.size != 2 * sizeof(uint32_t)) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Visible-count readback requires a two-word count buffer (buffer={:#x}, offset={}, active_bytes={}, allocation_bytes={}, required_bytes={})",
                lfs::rendering::vkHandleValue(count_buffer.buffer),
                count_buffer.offset,
                count_buffer.size,
                count_buffer.allocSize,
                2 * sizeof(uint32_t)),
            LFS_SOURCE_SITE_CURRENT());
    }
    // Never stomp an in-flight tagged copy: with the GPU a frame behind, the
    // re-record would reset the tag every frame and starve the count telemetry
    // and its loud overflow check.
    if (visible_count_readback_pending_ &&
        visible_count_readback_signal_ != VK_NULL_HANDLE)
        return;

    using lfs::rendering::vulkan::BufferUse;
    using lfs::rendering::vulkan::DeclaredAccess;

    // Plan TransferRead source + TransferWrite dst before the copy.
    const DeclaredAccess pre_copy[] = {
        {.buffer = &count_buffer, .use = BufferUse::TransferRead},
        {.buffer = &visible_count_readback_buffer_, .use = BufferUse::TransferWrite},
    };
    planTransfer(std::span{pre_copy});

    VkBufferCopy copy{};
    copy.srcOffset = buffers.visible_count.deviceBuffer.offset;
    copy.dstOffset = 0;
    copy.size = 2 * sizeof(uint32_t);
    validateBufferRange(count_buffer, 0, copy.size, "visible-count readback source");
    validateBufferRange(visible_count_readback_buffer_, 0, copy.size, "visible-count readback destination");
    if (vulkan_dispatch_.cmd_copy_buffer == nullptr) {
        lfs::rendering::throw_renderer_contract(
            "recordVisibleCountReadback requires VulkanDispatch::cmd_copy_buffer",
            LFS_SOURCE_SITE_CURRENT());
    }
    vulkan_dispatch_.cmd_copy_buffer(command_buffer,
                                     buffers.visible_count.deviceBuffer.buffer,
                                     visible_count_readback_buffer_.buffer,
                                     1,
                                     &copy);
    // Host coherence still needs fence/timeline wait at batch end (§3.2 G3).
    const DeclaredAccess host_read{
        .buffer = &visible_count_readback_buffer_,
        .use = BufferUse::HostRead,
    };
    planTransfer(std::span{&host_read, 1});
    visible_count_readback_pending_ = true;
    visible_count_readback_signal_ = VK_NULL_HANDLE;
    visible_count_readback_value_ = 0;
    visible_count_readback_num_splats_ = num_splats;
}

void VulkanGSRenderer::recordLodSelectionReadback(VulkanGSPipelineBuffers& buffers,
                                                  const size_t rendered_capacity) {
    // Fixed payload: 4 compact counts + protected ids + (chunk, priority)
    // pairs; independent of the logical chunk count.
    constexpr size_t kPayloadWords =
        4 + kLodCompactProtectedCap + 2 * kLodCompactMissCap;
    ensureLodSelectionReadback(kPayloadWords);
    if (buffers.lod_gpu_counts.deviceBuffer.buffer == VK_NULL_HANDLE ||
        buffers.lod_compact_counts.deviceBuffer.buffer == VK_NULL_HANDLE)
        return;

    using lfs::rendering::vulkan::BufferUse;
    using lfs::rendering::vulkan::DeclaredAccess;

    // Epic #1496 §3.2: plan TransferRead sources + TransferWrite dst before copies.
    // Readback buffer is untracked → conservative rows (expected).
    const DeclaredAccess pre_copy[] = {
        {.buffer = &buffers.lod_gpu_counts.deviceBuffer, .use = BufferUse::TransferRead},
        {.buffer = &buffers.lod_compact_counts.deviceBuffer, .use = BufferUse::TransferRead},
        {.buffer = &buffers.lod_compact_protected.deviceBuffer, .use = BufferUse::TransferRead},
        {.buffer = &buffers.lod_compact_misses.deviceBuffer, .use = BufferUse::TransferRead},
        {.buffer = &lod_selection_readback_buffer_, .use = BufferUse::TransferWrite},
    };
    planTransfer(std::span{pre_copy});

    const auto copy_region = [&](const _VulkanBuffer& src, const size_t dst_word,
                                 const size_t words) {
        VkBufferCopy copy{};
        copy.srcOffset = src.offset;
        copy.dstOffset = dst_word * sizeof(uint32_t);
        copy.size = words * sizeof(uint32_t);
        validateBufferRange(src, 0, copy.size, "LOD-selection readback source");
        validateBufferRange(lod_selection_readback_buffer_,
                            copy.dstOffset,
                            copy.size,
                            "LOD-selection readback destination");
        if (vulkan_dispatch_.cmd_copy_buffer == nullptr) {
            lfs::rendering::throw_renderer_contract(
                "recordLodSelectionReadback requires VulkanDispatch::cmd_copy_buffer",
                LFS_SOURCE_SITE_CURRENT());
        }
        vulkan_dispatch_.cmd_copy_buffer(command_buffer, src.buffer,
                                         lod_selection_readback_buffer_.buffer, 1, &copy);
    };
    copy_region(buffers.lod_gpu_counts.deviceBuffer, 0, 2);
    copy_region(buffers.lod_compact_counts.deviceBuffer, 2, 4);
    copy_region(buffers.lod_compact_protected.deviceBuffer, 6, kLodCompactProtectedCap);
    copy_region(buffers.lod_compact_misses.deviceBuffer, 6 + kLodCompactProtectedCap,
                2 * kLodCompactMissCap);

    // Host coherence still requires fence/timeline wait at endCommandBatch (§3.2 G3).
    const DeclaredAccess host_read{
        .buffer = &lod_selection_readback_buffer_,
        .use = BufferUse::HostRead,
    };
    planTransfer(std::span{&host_read, 1});

    lod_selection_readback_pending_ = true;
    lod_selection_readback_signal_ = VK_NULL_HANDLE;
    lod_selection_readback_value_ = 0;
    lod_selection_readback_capacity_ = rendered_capacity;
}

bool VulkanGSRenderer::invalidateReadbackBuffer(_VulkanBuffer& buffer, VkDeviceSize size) {
    validateBufferRange(buffer, 0, size, "invalidateReadbackBuffer");
    if (buffer.allocation == VK_NULL_HANDLE) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "invalidateReadbackBuffer requires a VMA-owned allocation (buffer={:#x}, allocation={:#x}, requested_bytes={}, allocation_bytes={}, label='{}')",
                lfs::rendering::vkHandleValue(buffer.buffer),
                lfs::rendering::vkHandleValue(buffer.allocation),
                size,
                buffer.allocSize,
                buffer.label ? buffer.label : "<unlabeled>"),
            LFS_SOURCE_SITE_CURRENT());
    }
    const VkResult result = vmaInvalidateAllocation(allocator, buffer.allocation, 0, size);
    if (result != VK_SUCCESS) {
        lfs::rendering::throw_vk_result(
            result,
            "vmaInvalidateAllocation",
            std::format(
                "vmaInvalidateAllocation failed for a VkSplat readback (buffer={:#x}, allocation={:#x}, requested_bytes={}, allocation_bytes={}, result={}({}))",
                lfs::rendering::vkHandleValue(buffer.buffer),
                lfs::rendering::vkHandleValue(buffer.allocation),
                size,
                buffer.allocSize,
                lfs::rendering::vkResultToString(result),
                static_cast<int>(result)),
            LFS_SOURCE_SITE_CURRENT());
    }
    return true;
}

void VulkanGSRenderer::initializeExternal(const std::map<std::string, std::string>& spirv_paths,
                                          VkInstance external_instance,
                                          VkPhysicalDevice external_physical_device,
                                          VkDevice external_device,
                                          VkQueue external_queue,
                                          uint32_t external_queue_family_index,
                                          VmaAllocator external_allocator,
                                          VkPipelineCache external_pipeline_cache,
                                          const bool supports_conditional_rendering,
                                          PFN_vkCmdBeginConditionalRenderingEXT begin_conditional_rendering,
                                          PFN_vkCmdEndConditionalRenderingEXT end_conditional_rendering) {
    destroyVisibleCountReadback();
    destroyLodSelectionReadback();
    destroyInstanceCountReadback();
    destroyInstanceGateReadback();
    VulkanGSPipeline::initializeExternal(
        external_instance,
        external_physical_device,
        external_device,
        external_queue,
        external_queue_family_index,
        external_allocator,
        external_pipeline_cache);
    supports_conditional_rendering_ = supports_conditional_rendering;
    vk_cmd_begin_conditional_rendering_ = begin_conditional_rendering;
    vk_cmd_end_conditional_rendering_ = end_conditional_rendering;
    if (supports_conditional_rendering_ &&
        (vk_cmd_begin_conditional_rendering_ == nullptr ||
         vk_cmd_end_conditional_rendering_ == nullptr)) {
        lfs::rendering::throw_renderer_contract(
            "Conditional rendering was enabled without both command entry points",
            LFS_SOURCE_SITE_CURRENT());
    }
    LOG_INFO("vksplat depth waves: {} slots ({})",
             supports_conditional_rendering_ ? HIGS_DEPTH_MAX_WAVES
                                             : HIGS_DEPTH_MAX_WAVES_FALLBACK,
             supports_conditional_rendering_ ? "conditional rendering"
                                             : "VK_EXT_conditional_rendering unavailable");

    createComputePipeline(pipeline_projection_forward, spirv_paths.at("projection_forward"));
    createComputePipeline(pipeline_projection_forward_3dgut, spirv_paths.at("projection_forward_3dgut"));
    createComputePipeline(pipeline_selection_mask, spirv_paths.at("selection_mask"));
    createComputePipeline(pipeline_selection_polygon_rasterize, spirv_paths.at("selection_polygon_rasterize"));
    createComputePipeline(pipeline_generate_keys_wave, spirv_paths.at("generate_keys_wave"));
    for (int i = 0; i < 2; ++i) {
        createComputePipeline(pipeline_compute_tile_ranges[i], spirv_paths.at("compute_tile_ranges"));
        createComputePipeline(pipeline_compute_tile_ranges_and_batch_counts[i],
                              spirv_paths.at("compute_tile_ranges_and_batch_counts"));
        createComputePipeline(pipeline_rasterize_forward[i], spirv_paths.at("rasterize_forward"));
        createComputePipeline(pipeline_rasterize_forward_3dgut[i], spirv_paths.at("rasterize_forward_3dgut"));
        createComputePipeline(pipeline_rasterize_forward_plain[i], spirv_paths.at("rasterize_forward_plain"));
        createComputePipeline(pipeline_rasterize_forward_3dgut_plain[i], spirv_paths.at("rasterize_forward_3dgut_plain"));
        createComputePipeline(pipeline_rasterize_forward_light[i],
                              spirv_paths.at("rasterize_forward_light"));
        createComputePipeline(pipeline_rasterize_forward_light_plain[i],
                              spirv_paths.at("rasterize_forward_light_plain"));
        createComputePipeline(pipeline_rasterize_forward_batches[i],
                              spirv_paths.at("rasterize_forward_batches"));
        createComputePipeline(pipeline_rasterize_forward_batches_plain[i],
                              spirv_paths.at("rasterize_forward_batches_plain"));
    }
    createComputePipeline(pipeline_tile_batch_descriptors, spirv_paths.at("tile_batch_descriptors"));
    createComputePipeline(pipeline_compose_tile_batches, spirv_paths.at("compose_tile_batches"));
    createComputePipeline(pipeline_compose_tile_batches_plain, spirv_paths.at("compose_tile_batches_plain"));
    createComputePipeline(pipeline_cumsum.single_pass, spirv_paths.at("cumsum_single_pass"));
    createComputePipeline(pipeline_cumsum.block_scan, spirv_paths.at("cumsum_block_scan"));
    createComputePipeline(pipeline_cumsum.scan_block_sums, spirv_paths.at("cumsum_scan_block_sums"));
    createComputePipeline(pipeline_cumsum.add_block_offsets, spirv_paths.at("cumsum_add_block_offsets"));
    createComputePipeline(pipeline_radix_histogram_clear, spirv_paths.at("radix_histogram_clear"));
    createComputePipeline(pipeline_expected_depth_finalize,
                          spirv_paths.at("expected_depth_finalize"));
    createComputePipeline(pipeline_sorting_indirect_1.upsweep, spirv_paths.at("radix_sort/upsweep_indirect"));
    createComputePipeline(pipeline_sorting_indirect_1.spine, spirv_paths.at("radix_sort/spine_indirect"));
    createComputePipeline(pipeline_sorting_indirect_1.downsweep, spirv_paths.at("radix_sort/downsweep_indirect"));
    createComputePipeline(pipeline_sorting_indirect_2.upsweep, spirv_paths.at("radix_sort/upsweep_indirect"));
    createComputePipeline(pipeline_sorting_indirect_2.spine, spirv_paths.at("radix_sort/spine_indirect"));
    createComputePipeline(pipeline_sorting_indirect_2.downsweep, spirv_paths.at("radix_sort/downsweep_indirect"));
    createComputePipeline(pipeline_apply_depth_ordering, spirv_paths.at("apply_depth_ordering"));
    createComputePipeline(pipeline_visible_flags, spirv_paths.at("visible_flags"));
    createComputePipeline(pipeline_prepare_visible_sort, spirv_paths.at("prepare_visible_sort"));
    createComputePipeline(pipeline_prepare_tile_sort, spirv_paths.at("prepare_tile_sort"));
    createComputePipeline(pipeline_compact_visible_primitives, spirv_paths.at("compact_visible_primitives"));
    createComputePipeline(pipeline_lod_map_indices, spirv_paths.at("lod_map_indices"));
    createComputePipeline(pipeline_lod_select_threshold, spirv_paths.at("lod_select_threshold"));
    if (spirv_paths.count("lod_compact_touch")) {
        createComputePipeline(pipeline_lod_compact_touch, spirv_paths.at("lod_compact_touch"));
    }

    // HiGS viewer chain pipelines are optional so trainer-side users of this
    // renderer (which pass only the legacy shader set) keep working.
    const auto create_optional = [&](auto& pipeline, const char* name) {
        const auto it = spirv_paths.find(name);
        if (it != spirv_paths.end()) {
            createComputePipeline(pipeline, it->second);
        }
    };
    create_optional(pipeline_cull_splats, "cull_splats");
    create_optional(pipeline_cull_prepare, "cull_prepare");
    create_optional(pipeline_projection_forward_survivors, "projection_forward_survivors");
    // Quant-pool projection variants exist only where the viewer registers
    // them; a quant pool with a missing pipeline is a hard dispatch error.
    create_optional(pipeline_projection_forward_quant, "projection_forward_quant");
    create_optional(pipeline_projection_forward_quant_3dgut, "projection_forward_quant_3dgut");
    create_optional(pipeline_projection_forward_quant_survivors, "projection_forward_quant_survivors");
    create_optional(pipeline_prepare_visible_chain, "prepare_visible_chain");
    create_optional(pipeline_copy_visible_indices, "copy_visible_indices");
    create_optional(pipeline_cumsum_indirect.block_scan, "cumsum_block_scan_indirect");
    create_optional(pipeline_cumsum_indirect.scan_block_sums, "cumsum_scan_block_sums_indirect");
    create_optional(pipeline_cumsum_indirect.add_block_offsets, "cumsum_add_block_offsets_indirect");
    create_optional(pipeline_prepare_tile_sort_visible, "prepare_tile_sort_visible");
    create_optional(pipeline_wave_partition, "wave_partition");
    create_optional(pipeline_wave_partition_visible, "wave_partition_visible");

    // The macro raster/compose pipelines store half4 partials, which needs
    // 16-bit storage + fp16 arithmetic. Without them the whole macro chain is
    // skipped (the viewer falls back to the legacy chain).
    {
        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceVulkan11Features f11{};
        f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        f11.pNext = &f12;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &f11;
        vkGetPhysicalDeviceFeatures2(external_physical_device, &f2);
        supports_float16_storage_ =
            f12.shaderFloat16 == VK_TRUE && f11.storageBuffer16BitAccess == VK_TRUE;
    }
    if (supports_float16_storage_) {
        create_optional(pipeline_macro_coverage, "macro_coverage");
        create_optional(pipeline_generate_macro_keys_wave, "generate_macro_keys_wave");
        create_optional(pipeline_macro_batch_prepare, "macro_batch_prepare");
        for (int i = 0; i < 2; ++i) {
            create_optional(pipeline_compute_macro_ranges[i], "compute_macro_ranges");
            create_optional(pipeline_macro_raster[i], "macro_raster");
            create_optional(pipeline_macro_raster_fp32[i], "macro_raster_fp32");
            create_optional(pipeline_macro_raster_overlays[i], "macro_raster_overlays");
            create_optional(pipeline_macro_compose[i], "macro_compose");
            create_optional(pipeline_macro_compose_overlays[i], "macro_compose_overlays");
        }
    }
}

void VulkanGSRenderer::executeMapLodIndices(const std::uint32_t lod_count,
                                            const std::uint32_t chunk_splats,
                                            const std::uint32_t invalid_page,
                                            VulkanGSPipelineBuffers& buffers,
                                            const _VulkanBuffer& chunk_to_page) {
    if (lod_count == 0 ||
        buffers.lod_logical_indices.deviceBuffer.buffer == VK_NULL_HANDLE ||
        chunk_to_page.buffer == VK_NULL_HANDLE) {
        return;
    }

    struct Uniforms {
        std::uint32_t lod_count;
        std::uint32_t chunk_splats;
        std::uint32_t invalid_page;
        std::uint32_t pad0;
    } map_uniforms{lod_count, chunk_splats, invalid_page, 0u};

    auto& out_indices = resizeDeviceBuffer(buffers.lod_indices, lod_count);

    // Tags from lod_map_indices.slang: logical/chunk_to_page StructuredBuffer (read),
    // out_indices RWStructuredBuffer (write). No post-barrier: projection still has
    // legacy pre-barriers on these outputs (§3.4.5).
    using lfs::rendering::vulkan::BufferUse;
    executeCompute(
        {{lod_count, 64}},
        &map_uniforms, sizeof(map_uniforms),
        pipeline_lod_map_indices,
        std::vector<TaggedBinding>{
            {buffers.lod_logical_indices.deviceBuffer, BufferUse::ComputeRead},
            {chunk_to_page, BufferUse::ComputeRead},
            {out_indices, BufferUse::ComputeWrite},
        });
}

void VulkanGSRenderer::executeSelectLodThreshold(const VulkanGSLodSelectUniforms& uniforms,
                                                 VulkanGSPipelineBuffers& buffers,
                                                 const _VulkanBuffer& node_bounds,
                                                 const _VulkanBuffer& node_links,
                                                 const _VulkanBuffer& chunk_to_page,
                                                 const _VulkanBuffer& page_age,
                                                 const _VulkanBuffer& page_frames,
                                                 const _VulkanBuffer& page_to_chunk) {
    if (uniforms.node_count == 0 ||
        uniforms.physical_node_count == 0 ||
        uniforms.output_capacity == 0 ||
        node_bounds.buffer == VK_NULL_HANDLE ||
        node_links.buffer == VK_NULL_HANDLE ||
        chunk_to_page.buffer == VK_NULL_HANDLE ||
        page_age.buffer == VK_NULL_HANDLE ||
        page_frames.buffer == VK_NULL_HANDLE ||
        page_to_chunk.buffer == VK_NULL_HANDLE) {
        return;
    }

    auto& counts = clearDeviceBuffer(buffers.lod_gpu_counts, 2);
    auto& out_indices = resizeDeviceBuffer(buffers.lod_gpu_indices, uniforms.output_capacity, true);
    auto& out_logical_indices = resizeDeviceBuffer(buffers.lod_gpu_logical_indices,
                                                   uniforms.output_capacity,
                                                   true);
    auto& out_weights = resizeDeviceBuffer(buffers.lod_gpu_weights, uniforms.output_capacity, true);
    auto& out_levels = resizeDeviceBuffer(buffers.lod_gpu_levels, uniforms.output_capacity, true);
    const size_t chunk_touch_count = std::max<size_t>(uniforms.logical_chunk_count, 1);
    auto& chunk_touch = clearDeviceBuffer(buffers.lod_chunk_touch, chunk_touch_count);

    // No sentinel fill of out_indices/out_logical_indices: projection gates on
    // the appended count in lod_gpu_counts[0], so entries past the valid prefix
    // are never read.
    // Tags from lod_select_threshold.slang bindings 0–11.
    using lfs::rendering::vulkan::BufferUse;
    executeCompute(
        {{uniforms.physical_node_count, 128}},
        &uniforms, sizeof(uniforms),
        pipeline_lod_select_threshold,
        std::vector<TaggedBinding>{
            {node_bounds, BufferUse::ComputeRead},
            {node_links, BufferUse::ComputeRead},
            {chunk_to_page, BufferUse::ComputeRead},
            {counts, BufferUse::ComputeReadWrite},
            {out_indices, BufferUse::ComputeWrite},
            {out_logical_indices, BufferUse::ComputeWrite},
            {out_weights, BufferUse::ComputeWrite},
            {chunk_touch, BufferUse::ComputeReadWrite},
            {out_levels, BufferUse::ComputeWrite},
            {page_age, BufferUse::ComputeRead},
            {page_frames, BufferUse::ComputeRead},
            {page_to_chunk, BufferUse::ComputeRead},
        });

    // Phase D: compact chunk_touch on the GPU so the readback and the CPU
    // request pass scale with the working set, not the logical chunk count.
    auto& compact_counts = clearDeviceBuffer(buffers.lod_compact_counts, 4);
    auto& compact_protected =
        resizeDeviceBuffer(buffers.lod_compact_protected, kLodCompactProtectedCap, true);
    auto& compact_misses =
        resizeDeviceBuffer(buffers.lod_compact_misses, 2 * kLodCompactMissCap, true);
    const VulkanGSLodCompactUniforms compact_uniforms{
        .chunk_count = uniforms.logical_chunk_count,
        .protected_capacity = kLodCompactProtectedCap,
        .miss_capacity = kLodCompactMissCap,
        .pad0 = 0,
    };
    // Tags from lod_compact_touch.slang: chunk_touch read; counts/protected/misses write.
    executeCompute(
        {{uniforms.logical_chunk_count, 256}},
        &compact_uniforms, sizeof(compact_uniforms),
        pipeline_lod_compact_touch,
        std::vector<TaggedBinding>{
            {chunk_touch, BufferUse::ComputeRead},
            {compact_counts, BufferUse::ComputeWrite},
            {compact_protected, BufferUse::ComputeWrite},
            {compact_misses, BufferUse::ComputeWrite},
        });
    recordLodSelectionReadback(buffers, uniforms.output_capacity);
}

void VulkanGSRenderer::executeProjectionForward(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    const _VulkanBuffer& transform_indices,
    const _VulkanBuffer& node_mask,
    const _VulkanBuffer& overlay_params,
    const _VulkanBuffer& model_transforms,
    size_t alloc_reserve,
    bool use_gut_projection,
    const _VulkanBuffer& lod_indices,
    const _VulkanBuffer& lod_logical_indices,
    const _VulkanBuffer& lod_levels,
    const _VulkanBuffer& lod_weights,
    const _VulkanBuffer& lod_counts) {
    PerfTimer::Timer<PerfTimer::ProjectionForward> timer(this);
    DEVICE_GUARD;

    using lfs::rendering::vulkan::BufferUse;
    using lfs::rendering::vulkan::DeclaredAccess;

    const size_t num_splats = static_cast<size_t>(uniforms.num_splats);
    size_t alloc_size = std::max(num_splats, alloc_reserve);

    // Two-stage sort: pre-fill primitive_depth_keys with 0xFFFFFFFFu so any
    // primitive that hits an early-return path inside the projection shader
    // keeps the max-key sentinel and sorts to the tail.
    auto& primitive_depth_keys =
        resizeDeviceBuffer(buffers.primitive_depth_keys, alloc_size);
    validateFillRange(primitive_depth_keys, 0, primitive_depth_keys.size, "primitive-depth sentinel fill");
    {
        const DeclaredAccess fill_access{
            .buffer = &primitive_depth_keys,
            .use = BufferUse::TransferWrite,
        };
        planTransfer(std::span{&fill_access, 1});
        if (vulkan_dispatch_.cmd_fill_buffer == nullptr) {
            lfs::rendering::throw_renderer_contract(
                "executeProjectionForward requires VulkanDispatch::cmd_fill_buffer",
                LFS_SOURCE_SITE_CURRENT());
        }
        vulkan_dispatch_.cmd_fill_buffer(command_buffer, primitive_depth_keys.buffer,
                                         primitive_depth_keys.offset, primitive_depth_keys.size,
                                         0xFFFFFFFFu);
    }

    // Optional LOD inputs: null handles use dummy bindings (same as legacy).
    // plan() skips VK_NULL_HANDLE; valid LOD buffers are tagged ComputeRead.
    const _VulkanBuffer lod_indices_binding =
        (lod_indices.buffer != VK_NULL_HANDLE) ? lod_indices : primitive_depth_keys;
    const _VulkanBuffer lod_logical_indices_binding =
        (lod_logical_indices.buffer != VK_NULL_HANDLE) ? lod_logical_indices : lod_indices_binding;
    const _VulkanBuffer lod_levels_binding =
        (lod_levels.buffer != VK_NULL_HANDLE) ? lod_levels : primitive_depth_keys;
    const _VulkanBuffer lod_weights_binding =
        (lod_weights.buffer != VK_NULL_HANDLE) ? lod_weights : primitive_depth_keys;
    const _VulkanBuffer lod_counts_binding =
        (lod_counts.buffer != VK_NULL_HANDLE) ? lod_counts : primitive_depth_keys;

    auto& tiles_touched = resizeDeviceBuffer(buffers.tiles_touched, alloc_size);
    auto& rect_tile_space = resizeDeviceBuffer(buffers.rect_tile_space, alloc_size);
    auto& radii = resizeDeviceBuffer(buffers.radii, alloc_size);
    auto& xy_vs = resizeDeviceBuffer(buffers.xy_vs, 2 * alloc_size);
    auto& depths = resizeDeviceBuffer(buffers.depths, alloc_size);
    auto& inv_cov = resizeDeviceBuffer(buffers.inv_cov_vs_opacity, 4 * alloc_size);
    auto& rgb = resizeDeviceBuffer(buffers.rgb, 3 * alloc_size);
    auto& overlay_flags = resizeDeviceBuffer(buffers.overlay_flags, alloc_size);

    // Binding order: catalog appendix "executeProjectionForward L1266 projection_buffers".
    // Tags: attrs/transform/node/overlay/model/LOD reads; projection outputs write;
    // primitive_depth_keys write (sentinel RMW after fill).
    std::vector<TaggedBinding> tagged = {
        {buffers.xyz_ws.deviceBuffer, BufferUse::ComputeRead},
        {buffers.sh0.deviceBuffer, BufferUse::ComputeRead},
        {buffers.shN.deviceBuffer, BufferUse::ComputeRead},
        {buffers.rotations.deviceBuffer, BufferUse::ComputeRead},
        {buffers.scaling_raw.deviceBuffer, BufferUse::ComputeRead},
        {buffers.opacity_raw.deviceBuffer, BufferUse::ComputeRead},
        {tiles_touched, BufferUse::ComputeWrite},
        {rect_tile_space, BufferUse::ComputeWrite},
        {radii, BufferUse::ComputeWrite},
        {xy_vs, BufferUse::ComputeWrite},
        {depths, BufferUse::ComputeWrite},
        {inv_cov, BufferUse::ComputeWrite},
        {rgb, BufferUse::ComputeWrite},
        {overlay_flags, BufferUse::ComputeWrite},
        {transform_indices, BufferUse::ComputeRead},
        {node_mask, BufferUse::ComputeRead},
        {overlay_params, BufferUse::ComputeRead},
        {model_transforms, BufferUse::ComputeRead},
        {primitive_depth_keys, BufferUse::ComputeWrite},
        {lod_indices_binding, BufferUse::ComputeRead},
        {lod_logical_indices_binding, BufferUse::ComputeRead},
        {lod_levels_binding, BufferUse::ComputeRead},
        {lod_weights_binding, BufferUse::ComputeRead},
        {lod_counts_binding, BufferUse::ComputeRead},
    };

    VulkanGSRendererUniforms projection_uniforms = uniforms;
    if (buffers.quant_pool) {
        projection_uniforms.lod_page_splats = buffers.pool_page_splats;
        tagged.push_back({buffers.page_frames.deviceBuffer, BufferUse::ComputeRead});
    }

    auto& pipeline = buffers.quant_pool
                         ? (use_gut_projection ? pipeline_projection_forward_quant_3dgut
                                               : pipeline_projection_forward_quant)
                         : (use_gut_projection ? pipeline_projection_forward_3dgut
                                               : pipeline_projection_forward);
    // Quant pipelines have 25 layouts; non-quant 24 — tagged size must match.
    executeCompute(
        {{num_splats, SUBGROUP_SIZE}},
        &projection_uniforms, sizeof(projection_uniforms),
        pipeline,
        tagged);
}

void VulkanGSRenderer::executeLegacyDepthWaves(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    const size_t armed,
    const int sort_bits,
    const _VulkanBuffer& selection_mask,
    const _VulkanBuffer& preview_mask,
    const _VulkanBuffer& selection_colors,
    const _VulkanBuffer& overlay_flags,
    const _VulkanBuffer& overlay_params,
    const _VulkanBuffer& transform_indices,
    const _VulkanBuffer& model_transforms,
    const bool use_gut_rasterization,
    const bool overlays_active,
    const bool predicate_waves) {
    PerfTimer::Timer<PerfTimer::RasterizeForward> timer(this);
    DEVICE_GUARD;

    if (armed == 0 || uniforms.sort_capacity != HIGS_DEPTH_WAVE_INSTANCES ||
        sort_bits <= 0 || sort_bits > 32) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Legacy depth waves require non-zero slots, fixed K, and sort bits in [1,32] (armed={}, uniform_capacity={}, K={}, sort_bits={})",
                armed,
                uniforms.sort_capacity,
                HIGS_DEPTH_WAVE_INSTANCES,
                sort_bits),
            LFS_SOURCE_SITE_CURRENT());
    }
    validateIndirectLayoutBuffer(buffers.depth_wave_dispatch.deviceBuffer,
                                 indirect::DepthWave::layout(armed),
                                 "legacy depth-wave consumer");
    if (buffers.wave_predicates.deviceBuffer.size < armed * sizeof(uint32_t)) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Legacy depth waves require one predicate per slot (armed={}, predicate_bytes={}, required_bytes={})",
                armed,
                buffers.wave_predicates.deviceBuffer.size,
                armed * sizeof(uint32_t)),
            LFS_SOURCE_SITE_CURRENT());
    }

    const size_t capacity = HIGS_DEPTH_WAVE_INSTANCES;
    // Logical counts: uniforms, dispatch bounds, heuristics, finalize push.
    const size_t num_tiles =
        static_cast<size_t>(uniforms.grid_height) * uniforms.grid_width;
    const size_t num_pixels =
        static_cast<size_t>(uniforms.image_height) * uniforms.image_width;
    if (num_tiles == 0 || num_pixels == 0)
        return;

    // Capacity for pixel/tile scratch: 64-px bucket (issue #1565). Indexing stays
    // logical (y * image_width + x); only resizeDeviceBuffer element counts use
    // alloc_*.
    const auto scratch_bucket = lfs::rendering::vulkan::viewportScratchBucket(
        uniforms.image_width, uniforms.image_height);
    const size_t alloc_pixels = scratch_bucket.alloc_pixels;
    const size_t alloc_tiles = scratch_bucket.alloc_tiles;
    if (scratch_bucket.alloc_w != scratch_bucket_alloc_w_ ||
        scratch_bucket.alloc_h != scratch_bucket_alloc_h_) {
        LOG_DEBUG(
            "vksplat.scratch.bucket logical={}x{} alloc={}x{} pixels_cap={} tiles_cap={}",
            uniforms.image_width,
            uniforms.image_height,
            scratch_bucket.alloc_w,
            scratch_bucket.alloc_h,
            alloc_pixels,
            alloc_tiles);
        scratch_bucket_alloc_w_ = scratch_bucket.alloc_w;
        scratch_bucket_alloc_h_ = scratch_bucket.alloc_h;
    }
    const auto resize_scratch = [this](auto& typed_buffer, const size_t elements) -> _VulkanBuffer& {
        auto& dev = typed_buffer.deviceBuffer;
        const size_t old_capacity = dev.capacity;
        auto& result = resizeDeviceBuffer(typed_buffer, elements);
        if (result.capacity != old_capacity) {
            LOG_DEBUG(
                "vksplat.scratch.realloc label={} old_bytes={} new_bytes={}",
                result.label != nullptr ? result.label : "<unlabeled>",
                old_capacity,
                result.capacity);
        }
        return result;
    };

    // This one-frame heuristic selects only the faster of two wave-correct
    // raster paths. It never sizes storage or changes the wave budget.
    const bool use_batched_raster =
        !use_gut_rasterization && !depth_capture_ &&
        buffers.num_indices >= kMinLoadBalancedRasterInstances &&
        buffers.num_indices / num_tiles >= kMinLoadBalancedAverageTileInstances;
    const size_t batch_capacity = denseTileBatchCapacity(capacity, alloc_tiles);

    // K/viewport allocations are established before conditional blocks.
    resizeDeviceBuffer(buffers.sorting_keys_1, capacity);
    resizeDeviceBuffer(buffers.sorting_keys_2, capacity);
    resizeDeviceBuffer(buffers.sorting_gauss_idx_1, capacity);
    resizeDeviceBuffer(buffers.sorting_gauss_idx_2, capacity);
    auto& tile_ranges = resize_scratch(buffers.tile_ranges, alloc_tiles + 1u);
    auto& pixel_state = resize_scratch(buffers.pixel_state, 4u * alloc_pixels);
    auto& pixel_depth = resize_scratch(buffers.pixel_depth, alloc_pixels);
    auto& pixel_depth_weight =
        resize_scratch(buffers.pixel_depth_weight, alloc_pixels);
    auto& n_contributors = resize_scratch(buffers.n_contributors, alloc_pixels);

    constexpr size_t kRadix = 256u;
    constexpr size_t kPartitionSize = 512u * 8u;
    const size_t radix_passes = _CEIL_DIV(static_cast<size_t>(sort_bits), size_t{8});
    resizeDeviceBuffer(buffers._sorting_histogram, radix_passes * kRadix);
    resizeDeviceBuffer(buffers._sorting_histogram_cumsum,
                       _CEIL_DIV(capacity, kPartitionSize) * kRadix);

    _VulkanBuffer batch_counts{};
    _VulkanBuffer batch_offsets{};
    _VulkanBuffer batch_descriptors{};
    _VulkanBuffer batch_dispatch{};
    _VulkanBuffer batch_pixel_state{};
    _VulkanBuffer batch_n_contributors{};
    if (use_batched_raster) {
        batch_counts = resize_scratch(buffers.tile_batch_counts, alloc_tiles);
        batch_offsets = resize_scratch(buffers.tile_batch_offsets, alloc_tiles);
        batch_descriptors =
            resize_scratch(buffers.tile_batch_descriptors, 4u * batch_capacity);
        batch_dispatch = resizeDeviceBuffer(
            buffers.tile_batch_dispatch_args,
            indirect::TileBatchDispatch::kLayout.word_count);
        validateIndirectLayoutBuffer(batch_dispatch,
                                     indirect::TileBatchDispatch::kLayout,
                                     "legacy tile-batch descriptor producer");
        batch_pixel_state = resize_scratch(
            buffers.tile_batch_pixel_state,
            4u * batch_capacity * TILE_WIDTH * TILE_HEIGHT);
        batch_n_contributors = resize_scratch(
            buffers.tile_batch_n_contributors,
            batch_capacity * TILE_WIDTH * TILE_HEIGHT);
        constexpr size_t kCumsumBlock = 1024u;
        // Cumsum spans the logical tile grid; capacity is still stable under
        // alloc_tiles when the bucket holds (alloc_tiles >= num_tiles).
        resizeDeviceBuffer(buffers._cumsum_blockSums,
                           std::max<size_t>(1u, _CEIL_DIV(alloc_tiles, kCumsumBlock)));
        resizeDeviceBuffer(
            buffers._cumsum_blockSums2,
            std::max<size_t>(1u,
                             _CEIL_DIV(_CEIL_DIV(alloc_tiles, kCumsumBlock),
                                       kCumsumBlock)));
    }

    using lfs::rendering::vulkan::BufferUse;
    using lfs::rendering::vulkan::DeclaredAccess;
    // Mega-hoist removed: each tagged/indirect dispatch plans its own hazards.

    const auto& wave_buffer = buffers.depth_wave_dispatch.deviceBuffer;
    const auto& predicate_buffer = buffers.wave_predicates.deviceBuffer;
    for (size_t wave = 0; wave < armed; ++wave) {
        const bool conditional = predicate_waves && supports_conditional_rendering_;
        // §3.4.5: ConditionalRenderingScope has no barrier site — plan ConditionalRead.
        if (conditional) {
            const DeclaredAccess pred{
                .buffer = &predicate_buffer,
                .use = BufferUse::ConditionalRead,
            };
            planTransfer(std::span{&pred, 1});
        }
        const ConditionalRenderingScope conditional_scope(
            *this,
            conditional,
            vk_cmd_begin_conditional_rendering_,
            vk_cmd_end_conditional_rendering_,
            predicate_buffer,
            wave * sizeof(uint32_t));

        // Inter-wave CSRW mega-barrier removed; per-dispatch plans cover reuse.

        VulkanGSRendererUniforms wave_uniforms = uniforms;
        wave_uniforms.depth_wave = static_cast<uint32_t>(wave);
        const auto record = bufferView(
            wave_buffer,
            indirect::byteOffset(indirect::DepthWave::recordWordOffset(wave)),
            indirect::byteSize(indirect::DepthWave::kRecordLayout));
        const auto count = bufferView(
            wave_buffer,
            indirect::byteOffset(indirect::DepthWave::countWordOffset(wave)),
            2u * sizeof(uint32_t));

        auto& unsorted_keys = buffers.unsorted_keys().deviceBuffer;
        auto& unsorted_indices = buffers.unsorted_gauss_idx().deviceBuffer;
        // pipeline_generate_keys_wave: 8 bindings (catalog).
        executeComputeIndirect(
            record,
            indirect::byteOffset(indirect::DepthWave::kKeygenWordOffset),
            &wave_uniforms,
            sizeof(wave_uniforms),
            pipeline_generate_keys_wave,
            std::vector<TaggedBinding>{
                {buffers.xy_vs.deviceBuffer, BufferUse::ComputeRead},
                {buffers.inv_cov_vs_opacity.deviceBuffer, BufferUse::ComputeRead},
                {buffers.rect_tile_space.deviceBuffer, BufferUse::ComputeRead},
                {buffers.index_buffer_offset.deviceBuffer, BufferUse::ComputeRead},
                {buffers.primitive_sort_indices.deviceBuffer, BufferUse::ComputeRead},
                {unsorted_keys, BufferUse::ComputeWrite},
                {unsorted_indices, BufferUse::ComputeWrite},
                {wave_buffer, BufferUse::ComputeRead},
            });
        executeSortIndirectCountImpl(wave_uniforms,
                                     buffers,
                                     sort_bits,
                                     count,
                                     record,
                                     capacity,
                                     indirect::DepthWave::kRecordLayout,
                                     indirect::DepthWave::kRadixWordOffset,
                                     "vksplat.render.record.sort_legacy_depth_wave",
                                     true);

        if (use_batched_raster) {
            executeComputeIndirect(
                record,
                indirect::byteOffset(indirect::DepthWave::kPerTileWordOffset),
                &wave_uniforms,
                sizeof(wave_uniforms),
                pipeline_compute_tile_ranges_and_batch_counts[buffers.is_unsorted_1],
                std::vector<TaggedBinding>{
                    {buffers.sorted_keys().deviceBuffer, BufferUse::ComputeRead},
                    {tile_ranges, BufferUse::ComputeWrite},
                    {count, BufferUse::ComputeRead},
                    {batch_counts, BufferUse::ComputeWrite},
                });
            executeCumsum(buffers,
                          buffers.tile_batch_counts,
                          buffers.tile_batch_offsets,
                          wave < HIGS_DEPTH_MAX_WAVES);
            executeComputeIndirect(
                record,
                indirect::byteOffset(indirect::DepthWave::kPerTileWordOffset),
                &wave_uniforms,
                sizeof(wave_uniforms),
                pipeline_tile_batch_descriptors,
                std::vector<TaggedBinding>{
                    {tile_ranges, BufferUse::ComputeRead},
                    {batch_offsets, BufferUse::ComputeRead},
                    {batch_descriptors, BufferUse::ComputeWrite},
                    {batch_dispatch, BufferUse::ComputeWrite},
                });

            auto& light_pipeline = overlays_active
                                       ? pipeline_rasterize_forward_light
                                       : pipeline_rasterize_forward_light_plain;
            executeComputeIndirect(
                record,
                indirect::byteOffset(indirect::DepthWave::kFullscreenWordOffset),
                &wave_uniforms,
                sizeof(wave_uniforms),
                light_pipeline[buffers.is_unsorted_1],
                std::vector<TaggedBinding>{
                    {buffers.sorted_gauss_idx().deviceBuffer, BufferUse::ComputeRead},
                    {tile_ranges, BufferUse::ComputeRead},
                    {buffers.xy_vs.deviceBuffer, BufferUse::ComputeRead},
                    {buffers.inv_cov_vs_opacity.deviceBuffer, BufferUse::ComputeRead},
                    {buffers.rgb.deviceBuffer, BufferUse::ComputeRead},
                    {buffers.depths.deviceBuffer, BufferUse::ComputeRead},
                    {pixel_state, BufferUse::ComputeReadWrite},
                    {pixel_depth, BufferUse::ComputeReadWrite},
                    {n_contributors, BufferUse::ComputeReadWrite},
                    {pixel_depth_weight, BufferUse::ComputeReadWrite},
                    {selection_mask, BufferUse::ComputeRead},
                    {preview_mask, BufferUse::ComputeRead},
                    {selection_colors, BufferUse::ComputeRead},
                    {overlay_flags, BufferUse::ComputeRead},
                    {overlay_params, BufferUse::ComputeRead},
                });

            std::vector<TaggedBinding> batch_bindings{
                {buffers.sorted_gauss_idx().deviceBuffer, BufferUse::ComputeRead},
                {batch_descriptors, BufferUse::ComputeRead},
                {buffers.xy_vs.deviceBuffer, BufferUse::ComputeRead},
                {buffers.inv_cov_vs_opacity.deviceBuffer, BufferUse::ComputeRead},
                {buffers.rgb.deviceBuffer, BufferUse::ComputeRead},
                {batch_pixel_state, BufferUse::ComputeReadWrite},
                {batch_n_contributors, BufferUse::ComputeReadWrite},
            };
            if (overlays_active) {
                batch_bindings.insert(batch_bindings.end(),
                                      {{selection_mask, BufferUse::ComputeRead},
                                       {preview_mask, BufferUse::ComputeRead},
                                       {selection_colors, BufferUse::ComputeRead},
                                       {overlay_flags, BufferUse::ComputeRead},
                                       {overlay_params, BufferUse::ComputeRead}});
            }
            auto& batch_pipeline = overlays_active
                                       ? pipeline_rasterize_forward_batches
                                       : pipeline_rasterize_forward_batches_plain;
            executeComputeIndirect(
                batch_dispatch,
                indirect::byteOffset(indirect::TileBatchDispatch::kRasterWordOffset),
                &wave_uniforms,
                sizeof(wave_uniforms),
                batch_pipeline[buffers.is_unsorted_1],
                batch_bindings);

            std::vector<TaggedBinding> compose_bindings{
                {buffers.sorted_gauss_idx().deviceBuffer, BufferUse::ComputeRead},
                {batch_descriptors, BufferUse::ComputeRead},
                {batch_offsets, BufferUse::ComputeRead},
                {buffers.xy_vs.deviceBuffer, BufferUse::ComputeRead},
                {buffers.inv_cov_vs_opacity.deviceBuffer, BufferUse::ComputeRead},
                {buffers.rgb.deviceBuffer, BufferUse::ComputeRead},
                {buffers.depths.deviceBuffer, BufferUse::ComputeRead},
                {batch_pixel_state, BufferUse::ComputeRead},
                {batch_n_contributors, BufferUse::ComputeRead},
                {pixel_state, BufferUse::ComputeReadWrite},
                {pixel_depth, BufferUse::ComputeReadWrite},
                {n_contributors, BufferUse::ComputeReadWrite},
            };
            if (overlays_active) {
                compose_bindings.insert(compose_bindings.end(),
                                        {{selection_mask, BufferUse::ComputeRead},
                                         {preview_mask, BufferUse::ComputeRead},
                                         {selection_colors, BufferUse::ComputeRead},
                                         {overlay_flags, BufferUse::ComputeRead},
                                         {overlay_params, BufferUse::ComputeRead}});
            }
            executeComputeIndirect(
                record,
                indirect::byteOffset(indirect::DepthWave::kFullscreenWordOffset),
                &wave_uniforms,
                sizeof(wave_uniforms),
                overlays_active ? pipeline_compose_tile_batches
                                : pipeline_compose_tile_batches_plain,
                compose_bindings);
        } else {
            executeComputeIndirect(
                record,
                indirect::byteOffset(indirect::DepthWave::kRangeWordOffset),
                &wave_uniforms,
                sizeof(wave_uniforms),
                pipeline_compute_tile_ranges[buffers.is_unsorted_1],
                std::vector<TaggedBinding>{
                    {buffers.sorted_keys().deviceBuffer, BufferUse::ComputeRead},
                    {tile_ranges, BufferUse::ComputeWrite},
                    {count, BufferUse::ComputeRead},
                });
        }

        if (!use_batched_raster) {
            if (use_gut_rasterization) {
                auto& gut_pipeline = overlays_active
                                         ? pipeline_rasterize_forward_3dgut
                                         : pipeline_rasterize_forward_3dgut_plain;
                executeComputeIndirect(
                    record,
                    indirect::byteOffset(indirect::DepthWave::kFullscreenWordOffset),
                    &wave_uniforms,
                    sizeof(wave_uniforms),
                    gut_pipeline[buffers.is_unsorted_1],
                    std::vector<TaggedBinding>{
                        {buffers.sorted_gauss_idx().deviceBuffer, BufferUse::ComputeRead},
                        {tile_ranges, BufferUse::ComputeRead},
                        {buffers.xy_vs.deviceBuffer, BufferUse::ComputeRead},
                        {buffers.inv_cov_vs_opacity.deviceBuffer, BufferUse::ComputeRead},
                        {buffers.rgb.deviceBuffer, BufferUse::ComputeRead},
                        {buffers.depths.deviceBuffer, BufferUse::ComputeRead},
                        {buffers.xyz_ws.deviceBuffer, BufferUse::ComputeRead},
                        {buffers.rotations.deviceBuffer, BufferUse::ComputeRead},
                        {buffers.scaling_raw.deviceBuffer, BufferUse::ComputeRead},
                        {buffers.opacity_raw.deviceBuffer, BufferUse::ComputeRead},
                        {pixel_state, BufferUse::ComputeReadWrite},
                        {pixel_depth, BufferUse::ComputeReadWrite},
                        {n_contributors, BufferUse::ComputeReadWrite},
                        {pixel_depth_weight, BufferUse::ComputeReadWrite},
                        {selection_mask, BufferUse::ComputeRead},
                        {preview_mask, BufferUse::ComputeRead},
                        {selection_colors, BufferUse::ComputeRead},
                        {overlay_flags, BufferUse::ComputeRead},
                        {overlay_params, BufferUse::ComputeRead},
                        {transform_indices, BufferUse::ComputeRead},
                        {model_transforms, BufferUse::ComputeRead},
                    });
            } else {
                auto& raster_pipeline = overlays_active
                                            ? pipeline_rasterize_forward
                                            : pipeline_rasterize_forward_plain;
                executeComputeIndirect(
                    record,
                    indirect::byteOffset(indirect::DepthWave::kFullscreenWordOffset),
                    &wave_uniforms,
                    sizeof(wave_uniforms),
                    raster_pipeline[buffers.is_unsorted_1],
                    std::vector<TaggedBinding>{
                        {buffers.sorted_gauss_idx().deviceBuffer, BufferUse::ComputeRead},
                        {tile_ranges, BufferUse::ComputeRead},
                        {buffers.xy_vs.deviceBuffer, BufferUse::ComputeRead},
                        {buffers.inv_cov_vs_opacity.deviceBuffer, BufferUse::ComputeRead},
                        {buffers.rgb.deviceBuffer, BufferUse::ComputeRead},
                        {buffers.depths.deviceBuffer, BufferUse::ComputeRead},
                        {pixel_state, BufferUse::ComputeReadWrite},
                        {pixel_depth, BufferUse::ComputeReadWrite},
                        {n_contributors, BufferUse::ComputeReadWrite},
                        {pixel_depth_weight, BufferUse::ComputeReadWrite},
                        {selection_mask, BufferUse::ComputeRead},
                        {preview_mask, BufferUse::ComputeRead},
                        {selection_colors, BufferUse::ComputeRead},
                        {overlay_flags, BufferUse::ComputeRead},
                        {overlay_params, BufferUse::ComputeRead},
                    });
            }
        }
    }

    if (uniforms.expected_far > 0.0f) {
        const uint32_t finalize_uniforms = static_cast<uint32_t>(num_pixels);
        executeCompute({{num_pixels, 256u}},
                       &finalize_uniforms,
                       sizeof(finalize_uniforms),
                       pipeline_expected_depth_finalize,
                       std::vector<TaggedBinding>{
                           {pixel_depth, BufferUse::ComputeReadWrite},
                           {pixel_depth_weight, BufferUse::ComputeRead},
                       });
    }
}

void VulkanGSRenderer::executeSelectionMask(
    const VulkanGSSelectionMaskUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    const _VulkanBuffer& transform_indices,
    const _VulkanBuffer& node_mask,
    const _VulkanBuffer& primitives,
    const _VulkanBuffer& model_transforms,
    const _VulkanBuffer& selection_out,
    const _VulkanBuffer& polygon_mask,
    const _VulkanBuffer& ring_pick_out) {
    DEVICE_GUARD;

    using lfs::rendering::vulkan::BufferUse;
    using lfs::rendering::vulkan::DeclaredAccess;

    // Tags from selection_mask.slang bindings 0–10.
    const size_t num_words = _CEIL_DIV(static_cast<size_t>(uniforms.num_splats), 4);
    executeCompute(
        {{num_words, SUBGROUP_SIZE}},
        &uniforms, sizeof(uniforms),
        pipeline_selection_mask,
        std::vector<TaggedBinding>{
            {buffers.xyz_ws.deviceBuffer, BufferUse::ComputeRead},
            {transform_indices, BufferUse::ComputeRead},
            {node_mask, BufferUse::ComputeRead},
            {primitives, BufferUse::ComputeRead},
            {model_transforms, BufferUse::ComputeRead},
            {buffers.rotations.deviceBuffer, BufferUse::ComputeRead},
            {buffers.scaling_raw.deviceBuffer, BufferUse::ComputeRead},
            {selection_out, BufferUse::ComputeWrite},
            {polygon_mask, BufferUse::ComputeRead},
            {buffers.opacity_raw.deviceBuffer, BufferUse::ComputeRead},
            {ring_pick_out, BufferUse::ComputeWrite},
        });

    // Handoff: host/CUDA download consumers lack their own barrier site (§3.4.5).
    // Catalog post was COMPUTE_SHADER_WRITE → TRANSFER_READ.
    const DeclaredAccess transfer_handoff[] = {
        {.buffer = &selection_out, .use = BufferUse::TransferRead},
        {.buffer = &ring_pick_out, .use = BufferUse::TransferRead},
    };
    planTransfer(std::span{transfer_handoff});
}

void VulkanGSRenderer::executeSelectionPolygonRasterize(
    const VulkanGSSelectionPolygonRasterizeUniforms& uniforms,
    const _VulkanBuffer& polygon_vertices,
    const _VulkanBuffer& polygon_mask) {
    DEVICE_GUARD;

    using lfs::rendering::vulkan::BufferUse;

    // Tags from selection_polygon_rasterize.slang: vertices read, coverage_mask write.
    // Post CSR deleted: selection_mask plans ComputeRead on polygon_mask (§3.4.5 co-migrated).
    constexpr size_t kBlockXY = 8;
    executeCompute(
        {{static_cast<size_t>(uniforms.aabb_w), kBlockXY},
         {static_cast<size_t>(uniforms.aabb_h), kBlockXY}},
        &uniforms, sizeof(uniforms),
        pipeline_selection_polygon_rasterize,
        std::vector<TaggedBinding>{
            {polygon_vertices, BufferUse::ComputeRead},
            {polygon_mask, BufferUse::ComputeWrite},
        });
}

void VulkanGSRenderer::executeCumsum(
    VulkanGSPipelineBuffers& buffers,
    Buffer<int32_t>& input_buffer,
    Buffer<int32_t>& output_buffer,
    const bool record_timestamps) {
    std::optional<PerfTimer::Timer<PerfTimer::_Cumsum>> timer;
    if (record_timestamps) {
        timer.emplace(this);
    }
    DEVICE_GUARD;

    using lfs::rendering::vulkan::BufferUse;

    size_t num_elements = input_buffer.deviceSize();
    const size_t block_0 = 1024;
    const size_t block_limit = deviceInfo.subgroupSize * deviceInfo.subgroupSize * deviceInfo.subgroupSize;
    const size_t block = std::min(block_0, block_limit);

    // Tags from cumsum.slang: input read, output write, blockSums rw.
    auto execute_cumsum_phase = [&](size_t active_elements,
                                    size_t threads_per_group,
                                    _ComputePipeline& pipeline,
                                    const std::vector<TaggedBinding>& phase_bindings) {
        uint32_t phase_uniforms[1] = {static_cast<uint32_t>(active_elements)};
        executeCompute(
            {{active_elements, threads_per_group}},
            phase_uniforms,
            sizeof(uint32_t),
            pipeline,
            phase_bindings);
    };

    resizeDeviceBuffer(output_buffer, num_elements);

    // A scan phase writes gid < active_elements into its bound storage. A backing
    // smaller than that is silent out-of-bounds GPU writes with robustness off
    // (caught live by GPU-AV as VUID 06936), so pin the contract host-side.
    const auto require_backing = [](const _VulkanBuffer& b, const size_t needed_elements,
                                    const char* role) {
        const size_t needed_bytes = needed_elements * sizeof(int32_t);
        if (b.buffer == VK_NULL_HANDLE || b.allocSize < needed_bytes || b.size < needed_bytes) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "VkSplat cumsum {} backing is smaller than the scan (label='{}', buffer={:#x}, alloc_bytes={}, capacity_bytes={}, active_bytes={}, needed_bytes={}, elements={})",
                    role,
                    b.label ? b.label : "?",
                    lfs::rendering::vkHandleValue(b.buffer),
                    b.allocSize,
                    b.capacity,
                    b.size,
                    needed_bytes,
                    needed_elements),
                LFS_SOURCE_SITE_CURRENT());
        }
    };
    require_backing(input_buffer.deviceBuffer, num_elements, "input");
    require_backing(output_buffer.deviceBuffer, num_elements, "output");

    if (num_elements <= block_0) {
        execute_cumsum_phase(
            num_elements, block_0,
            pipeline_cumsum.single_pass,
            {
                {input_buffer.deviceBuffer, BufferUse::ComputeRead},
                {output_buffer.deviceBuffer, BufferUse::ComputeWrite},
            });
    }

    else if (num_elements <= block * block) {
        const size_t num_blocks = _CEIL_DIV(num_elements, block);
        resizeDeviceBuffer(buffers._cumsum_blockSums, num_blocks, true);
        require_backing(buffers._cumsum_blockSums.deviceBuffer, num_blocks, "block_sums");

        execute_cumsum_phase(
            num_elements, block,
            pipeline_cumsum.block_scan,
            {
                {input_buffer.deviceBuffer, BufferUse::ComputeRead},
                {output_buffer.deviceBuffer, BufferUse::ComputeWrite},
                {buffers._cumsum_blockSums.deviceBuffer, BufferUse::ComputeReadWrite},
            });

        execute_cumsum_phase(
            num_blocks, block,
            pipeline_cumsum.scan_block_sums,
            {
                {input_buffer.deviceBuffer, BufferUse::ComputeRead},
                {output_buffer.deviceBuffer, BufferUse::ComputeWrite},
                {buffers._cumsum_blockSums.deviceBuffer, BufferUse::ComputeReadWrite},
            });

        execute_cumsum_phase(
            num_elements, block,
            pipeline_cumsum.add_block_offsets,
            {
                {input_buffer.deviceBuffer, BufferUse::ComputeRead},
                {output_buffer.deviceBuffer, BufferUse::ComputeWrite},
                {buffers._cumsum_blockSums.deviceBuffer, BufferUse::ComputeReadWrite},
            });
    }

    else if (num_elements <= block * block * block) {
        const size_t num_elements_1 = _CEIL_DIV(num_elements, block);
        const size_t num_elements_2 = _CEIL_DIV(num_elements_1, block);
        resizeDeviceBuffer(buffers._cumsum_blockSums, num_elements_1, true);
        resizeDeviceBuffer(buffers._cumsum_blockSums2, num_elements_2, true);
        require_backing(buffers._cumsum_blockSums.deviceBuffer, num_elements_1, "block_sums");
        require_backing(buffers._cumsum_blockSums2.deviceBuffer, num_elements_2, "block_sums2");

        execute_cumsum_phase(
            num_elements, block,
            pipeline_cumsum.block_scan,
            {
                {input_buffer.deviceBuffer, BufferUse::ComputeRead},
                {output_buffer.deviceBuffer, BufferUse::ComputeWrite},
                {buffers._cumsum_blockSums.deviceBuffer, BufferUse::ComputeReadWrite},
            });

        execute_cumsum_phase(
            num_elements_1, block,
            pipeline_cumsum.block_scan,
            {
                {buffers._cumsum_blockSums.deviceBuffer, BufferUse::ComputeRead},
                {buffers._cumsum_blockSums.deviceBuffer, BufferUse::ComputeWrite},
                {buffers._cumsum_blockSums2.deviceBuffer, BufferUse::ComputeReadWrite},
            });

        execute_cumsum_phase(
            num_elements_2, block,
            pipeline_cumsum.scan_block_sums,
            {
                {buffers._cumsum_blockSums.deviceBuffer, BufferUse::ComputeRead},
                {buffers._cumsum_blockSums.deviceBuffer, BufferUse::ComputeWrite},
                {buffers._cumsum_blockSums2.deviceBuffer, BufferUse::ComputeReadWrite},
            });

        execute_cumsum_phase(
            num_elements_1, block,
            pipeline_cumsum.add_block_offsets,
            {
                {buffers._cumsum_blockSums.deviceBuffer, BufferUse::ComputeRead},
                {buffers._cumsum_blockSums.deviceBuffer, BufferUse::ComputeWrite},
                {buffers._cumsum_blockSums2.deviceBuffer, BufferUse::ComputeReadWrite},
            });

        execute_cumsum_phase(
            num_elements, block,
            pipeline_cumsum.add_block_offsets,
            {
                {input_buffer.deviceBuffer, BufferUse::ComputeRead},
                {output_buffer.deviceBuffer, BufferUse::ComputeWrite},
                {buffers._cumsum_blockSums.deviceBuffer, BufferUse::ComputeReadWrite},
            });
    }

    // can't reasonably expect more than 1G splats
    // although there may be more than 1G sorting indices
    else {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "VkSplat cumsum exceeds the supported three-level scan (element_count={}, block_size={}, level1_groups={}, level2_groups={}, max_level2_groups={})",
                num_elements,
                block,
                _CEIL_DIV(num_elements, block),
                _CEIL_DIV(_CEIL_DIV(num_elements, block), block),
                block),
            LFS_SOURCE_SITE_CURRENT());
    }
}

void VulkanGSRenderer::executeCalculateIndexBufferOffset(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers) {
    PerfTimer::Timer<PerfTimer::CalculateIndexBufferOffset> timer(this);

    const size_t num_elements = static_cast<size_t>(uniforms.num_splats);
    if (num_elements == 0) {
        buffers.num_indices = 0;
        return;
    }

    // Cumsum on tiles_touched_depth_ordered (output of executeApplyDepthOrdering)
    // so index_buffer_offset[depth_rank] gives the contiguous offset interval
    // for the primitive at depth rank `depth_rank`. Matches the gsplat_fwd CUDA
    // reference (cub::DeviceScan::ExclusiveSum on the reordered offsets array).
    executeCumsum(
        buffers,
        buffers.tiles_touched_depth_ordered,
        buffers.index_buffer_offset);

    executePrepareTileSort(uniforms, buffers);
}

void VulkanGSRenderer::executePrepareTileSort(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers) {
    PerfTimer::Timer<PerfTimer::PrepareTileSort> timer(this);
    [[maybe_unused]] auto cpu_timer =
        timeCpuStage("vksplat.render.record.executePrepareTileSort");
    DEVICE_GUARD;

    using lfs::rendering::vulkan::BufferUse;
    using lfs::rendering::vulkan::DeclaredAccess;

    resizeDeviceBuffer(buffers.tile_sort_count, 1);
    if (buffers.tile_sort_count.deviceBuffer.size != sizeof(uint32_t)) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "prepare_tile_sort count buffer must contain exactly one uint32 word (buffer={:#x}, active_bytes={}, allocation_bytes={}, required_bytes={})",
                lfs::rendering::vkHandleValue(buffers.tile_sort_count.deviceBuffer.buffer),
                buffers.tile_sort_count.deviceBuffer.size,
                buffers.tile_sort_count.deviceBuffer.allocSize,
                sizeof(uint32_t)),
            LFS_SOURCE_SITE_CURRENT());
    }
    const uint32_t num_splats = uniforms.num_splats;
    // Shader indexes index_buffer_offset[num_splats-1]; dual-source with cumsum size.
    if (num_splats > 0 &&
        buffers.index_buffer_offset.deviceSize() < static_cast<size_t>(num_splats)) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "prepare_tile_sort requires index_buffer_offset covering uniforms.num_splats (num_splats={}, device_elements={}, buffer={:#x})",
                num_splats,
                buffers.index_buffer_offset.deviceSize(),
                lfs::rendering::vkHandleValue(buffers.index_buffer_offset.deviceBuffer.buffer)),
            LFS_SOURCE_SITE_CURRENT());
    }

    // Tags from prepare_tile_sort.slang (non-visible): offset read, count write.
    executeCompute(
        {{1, 1}},
        &num_splats, sizeof(num_splats),
        pipeline_prepare_tile_sort,
        std::vector<TaggedBinding>{
            {buffers.index_buffer_offset.deviceBuffer, BufferUse::ComputeRead},
            {buffers.tile_sort_count.deviceBuffer, BufferUse::ComputeWrite},
        });
    // Handoff for wave/host consumers that still use TRANSFER_COMPUTE read scopes.
    const DeclaredAccess handoff[] = {
        {.buffer = &buffers.tile_sort_count.deviceBuffer, .use = BufferUse::TransferRead},
        {.buffer = &buffers.tile_sort_count.deviceBuffer, .use = BufferUse::ComputeRead},
    };
    planTransfer(std::span{handoff});
}

void VulkanGSRenderer::executeSortIndirectCount(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    int num_bits,
    const _VulkanBuffer& count_buffer,
    const _VulkanBuffer& dispatch_args_buffer,
    size_t capacity,
    const indirect::Layout& dispatch_layout,
    const size_t radix_word_offset) {
    PerfTimer::Timer<PerfTimer::SortVisiblePrimitives> timer(this);
    executeSortIndirectCountImpl(uniforms,
                                 buffers,
                                 num_bits,
                                 count_buffer,
                                 dispatch_args_buffer,
                                 capacity,
                                 dispatch_layout,
                                 radix_word_offset,
                                 "vksplat.render.record.sort_primitive_indirect",
                                 false);
}

void VulkanGSRenderer::executeSortIndirectCountImpl(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    int num_bits,
    const _VulkanBuffer& count_buffer,
    const _VulkanBuffer& dispatch_args_buffer,
    size_t capacity,
    const indirect::Layout& dispatch_layout,
    const size_t radix_word_offset,
    const char* cpu_timer_prefix,
    const bool wave_barriers_hoisted) {
    if (capacity == 0)
        return;
    if (radix_word_offset > dispatch_layout.word_count ||
        dispatch_layout.word_count - radix_word_offset < indirect::kCommandWordCount) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Indirect radix-sort layout must contain a complete VkDispatchIndirectCommand at its named radix offset (layout_constant='{}', layout_words={}, radix_word_offset={}, command_words={})",
                dispatch_layout.word_count_constant,
                dispatch_layout.word_count,
                radix_word_offset,
                indirect::kCommandWordCount),
            LFS_SOURCE_SITE_CURRENT());
    }
    validateIndirectLayoutBuffer(dispatch_args_buffer,
                                 dispatch_layout,
                                 "indirect radix sort consumer");
    if (capacity > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        count_buffer.size != 2 * sizeof(uint32_t) ||
        num_bits <= 0 || num_bits > 32) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Indirect radix sort requires a two-word count, bit count in [1, 32], and INT32-bounded capacity (capacity={}, int32_max={}, count_buffer={:#x}, count_bytes={}, dispatch_buffer={:#x}, dispatch_bytes={}, dispatch_layout='{}', dispatch_words={}, radix_word_offset={}, num_bits={})",
                capacity,
                std::numeric_limits<int32_t>::max(),
                lfs::rendering::vkHandleValue(count_buffer.buffer),
                count_buffer.size,
                lfs::rendering::vkHandleValue(dispatch_args_buffer.buffer),
                dispatch_args_buffer.size,
                dispatch_layout.word_count_constant,
                dispatch_layout.word_count,
                radix_word_offset,
                num_bits),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (capacity != buffers.unsorted_keys().deviceSize() ||
        capacity != buffers.unsorted_gauss_idx().deviceSize()) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Indirect radix sort capacity must match both input arrays (capacity={}, key_elements={}, value_elements={}, key_bytes={}, value_bytes={})",
                capacity,
                buffers.unsorted_keys().deviceSize(),
                buffers.unsorted_gauss_idx().deviceSize(),
                buffers.unsorted_keys().deviceBuffer.size,
                buffers.unsorted_gauss_idx().deviceBuffer.size),
            LFS_SOURCE_SITE_CURRENT());
    }

    const auto timer_name = [cpu_timer_prefix](const char* suffix) {
        return std::string(cpu_timer_prefix) + suffix;
    };

    const int RADIX = 256;
    const int WORKGROUP_SIZE = 512;
    const int PARTITION_DIVISION = 8;
    const int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;

    auto& globalHistogram = buffers._sorting_histogram;
    auto& partitionHistogram = buffers._sorting_histogram_cumsum;

    const size_t num_parts_capacity = _CEIL_DIV(capacity, PARTITION_SIZE);

    int max_nonzero_bit = 8 * sizeof(sortingKey_t);
    if (num_bits == -1 && sizeof(sortingKey_t) == 8) {
        int32_t num_tiles = (int32_t)(uniforms.grid_height * uniforms.grid_width);
        max_nonzero_bit = 23;
        int32_t temp = num_tiles;
        while (temp)
            temp >>= 1, max_nonzero_bit++;
    } else if (num_bits >= 0)
        max_nonzero_bit = num_bits;
    int num_passes = _CEIL_DIV(max_nonzero_bit, 8);

    {
        [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".resize_buffers"));
        resizeDeviceBuffer(globalHistogram, num_passes * RADIX);
        resizeDeviceBuffer(partitionHistogram, num_parts_capacity * RADIX);
        resizeDeviceBuffer(buffers.sorted_keys(), capacity);
        resizeDeviceBuffer(buffers.sorted_gauss_idx(), capacity);
    }

    DEVICE_GUARD;
    using lfs::rendering::vulkan::BufferUse;

    {
        [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".clear_histogram"));
        // Tags from radix_histogram_clear.slang: both histograms write.
        // Upsweep's ComputeRead on keys plans RAW vs prior keygen/pass.
        // When wave_barriers_hoisted, also plan key reads here so the wave
        // keygen→radix edge is not lost if the caller only hoisted wave reuse.
        const uint32_t clear_uniforms[2]{
            static_cast<uint32_t>(num_passes * RADIX),
            static_cast<uint32_t>(num_parts_capacity * RADIX),
        };
        executeCompute({{1, 1}},
                       clear_uniforms,
                       sizeof(clear_uniforms),
                       pipeline_radix_histogram_clear,
                       std::vector<TaggedBinding>{
                           {globalHistogram.deviceBuffer, BufferUse::ComputeWrite},
                           {partitionHistogram.deviceBuffer, BufferUse::ComputeWrite},
                       });
        if (wave_barriers_hoisted) {
            const lfs::rendering::vulkan::DeclaredAccess key_reads[] = {
                {.buffer = &buffers.unsorted_keys().deviceBuffer,
                 .use = BufferUse::ComputeRead},
                {.buffer = &buffers.unsorted_gauss_idx().deviceBuffer,
                 .use = BufferUse::ComputeRead},
            };
            planTransfer(std::span{key_reads});
        }
    }
    // count/dispatch → compute/indirect: planned by upsweep (IndirectRead implicit
    // + count ComputeRead). When wave_barriers_hoisted, caller already ordered them.

    for (int pass = 0; 8 * pass < max_nonzero_bit; pass++) {
        auto& pipeline_sorting = buffers.is_unsorted_1 ? pipeline_sorting_indirect_1
                                                       : pipeline_sorting_indirect_2;

        uint32_t sort_uniforms[2];
        sort_uniforms[0] = static_cast<uint32_t>(pass);
        sort_uniforms[1] = 0;

        // Pass ping-pong: prior downsweep wrote sorted_* which become unsorted_*
        // after the flip at end of pass. Tags on upsweep/downsweep plan the RAW/WAW.
        {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_upsweep"));
            // upsweep.comp: keys R, global RW, partition W, count R
            executeComputeIndirect(
                dispatch_args_buffer,
                indirect::byteOffset(radix_word_offset),
                sort_uniforms, 2 * sizeof(int32_t),
                pipeline_sorting.upsweep,
                std::vector<TaggedBinding>{
                    {buffers.unsorted_keys().deviceBuffer, BufferUse::ComputeRead},
                    {globalHistogram.deviceBuffer, BufferUse::ComputeReadWrite},
                    {partitionHistogram.deviceBuffer, BufferUse::ComputeWrite},
                    {count_buffer, BufferUse::ComputeRead},
                });
        }

        {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_spine"));
            // spine.comp: global RW, partition RW, count R
            executeCompute(
                {{RADIX, 1}},
                sort_uniforms, 2 * sizeof(int32_t),
                pipeline_sorting.spine,
                std::vector<TaggedBinding>{
                    {globalHistogram.deviceBuffer, BufferUse::ComputeReadWrite},
                    {partitionHistogram.deviceBuffer, BufferUse::ComputeReadWrite},
                    {count_buffer, BufferUse::ComputeRead},
                });
        }

        {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_downsweep"));
            // downsweep.comp: hist R, keys/values in R, keys/values out W, count R
            executeComputeIndirect(
                dispatch_args_buffer,
                indirect::byteOffset(radix_word_offset),
                sort_uniforms, 2 * sizeof(int32_t),
                pipeline_sorting.downsweep,
                std::vector<TaggedBinding>{
                    {globalHistogram.deviceBuffer, BufferUse::ComputeRead},
                    {partitionHistogram.deviceBuffer, BufferUse::ComputeRead},
                    {buffers.unsorted_keys().deviceBuffer, BufferUse::ComputeRead},
                    {buffers.unsorted_gauss_idx().deviceBuffer, BufferUse::ComputeRead},
                    {buffers.sorted_keys().deviceBuffer, BufferUse::ComputeWrite},
                    {buffers.sorted_gauss_idx().deviceBuffer, BufferUse::ComputeWrite},
                    {count_buffer, BufferUse::ComputeRead},
                });
        }

        buffers.is_unsorted_1 = !buffers.is_unsorted_1;
    }
    buffers.is_unsorted_1 = !buffers.is_unsorted_1;
}

void VulkanGSRenderer::executeSortPrimitivesByDepth(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers) {
    PerfTimer::Timer<PerfTimer::SortPrimitivesByDepth> timer(this);

    const size_t num_splats = static_cast<size_t>(uniforms.num_splats);
    if (num_splats == 0)
        return;

    DEVICE_GUARD;

    // Stage 1 follows the old CUDA path: reject/projection work stays N-wide,
    // but the expensive depth radix sort only sees compact visible primitives.
    // The ping-pong sort buffers still have N capacity so the GPU scatter cannot
    // overflow; the indirect sort count comes from the visible-prefix tail.
    _VulkanBuffer* unsorted_keys = nullptr;
    _VulkanBuffer* unsorted_idx = nullptr;
    {
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.ensure_buffers");
        unsorted_keys = &resizeDeviceBuffer(buffers.unsorted_keys(), num_splats);
        unsorted_idx = &resizeDeviceBuffer(buffers.unsorted_gauss_idx(), num_splats);
        resizeDeviceBuffer(buffers.visible_flags, num_splats);
        resizeDeviceBuffer(buffers.visible_count, 2);
        resizeDeviceBuffer(buffers.visible_sort_dispatch_args,
                           indirect::VisibleSortDispatch::kLayout.word_count);
        validateIndirectLayoutBuffer(buffers.visible_sort_dispatch_args.deviceBuffer,
                                     indirect::VisibleSortDispatch::kLayout,
                                     "prepare_visible_sort producer");
    }

    struct VisibleUniforms {
        uint32_t num_splats;
        uint32_t pad0, pad1, pad2;
    } visible_uniforms{static_cast<uint32_t>(num_splats), 0, 0, 0};

    using lfs::rendering::vulkan::BufferUse;
    using lfs::rendering::vulkan::DeclaredAccess;

    {
        PerfTimer::Timer<PerfTimer::BuildVisibleFlags> gpu_timer(this);
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.build_visible_flags");
        executeCompute(
            {{num_splats, 64}},
            &visible_uniforms, sizeof(visible_uniforms),
            pipeline_visible_flags,
            std::vector<TaggedBinding>{
                {buffers.tiles_touched.deviceBuffer, BufferUse::ComputeRead},
                {buffers.visible_flags.deviceBuffer, BufferUse::ComputeWrite},
            });
    }

    {
        PerfTimer::Timer<PerfTimer::VisiblePrefix> gpu_timer(this);
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.visible_prefix");
        executeCumsum(buffers, buffers.visible_flags, buffers.visible_prefix);
    }

    struct PrepareUniforms {
        uint32_t num_splats;
        uint32_t sort_partition_size;
        uint32_t pad0, pad1;
    } prepare_uniforms{static_cast<uint32_t>(num_splats), 512u * 8u, 0, 0};

    {
        PerfTimer::Timer<PerfTimer::PrepareVisibleSort> gpu_timer(this);
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.prepare_visible_sort");
        // Shader indexes visible_prefix[num_splats-1]; dual-source with cumsum size.
        if (num_splats > 0 && buffers.visible_prefix.deviceSize() < num_splats) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "prepare_visible_sort requires visible_prefix covering uniforms.num_splats (num_splats={}, device_elements={}, buffer={:#x})",
                    num_splats,
                    buffers.visible_prefix.deviceSize(),
                    lfs::rendering::vkHandleValue(buffers.visible_prefix.deviceBuffer.buffer)),
                LFS_SOURCE_SITE_CURRENT());
        }
        executeCompute(
            {{1, 1}},
            &prepare_uniforms, sizeof(prepare_uniforms),
            pipeline_prepare_visible_sort,
            std::vector<TaggedBinding>{
                {buffers.visible_prefix.deviceBuffer, BufferUse::ComputeRead},
                {buffers.visible_count.deviceBuffer, BufferUse::ComputeWrite},
                {buffers.visible_sort_dispatch_args.deviceBuffer, BufferUse::ComputeWrite},
            });
        // Handoff for readback TransferRead + sort count ComputeRead.
        const DeclaredAccess count_handoff[] = {
            {.buffer = &buffers.visible_count.deviceBuffer, .use = BufferUse::TransferRead},
            {.buffer = &buffers.visible_count.deviceBuffer, .use = BufferUse::ComputeRead},
        };
        planTransfer(std::span{count_handoff});
        recordVisibleCountReadback(buffers, num_splats);
    }

    {
        PerfTimer::Timer<PerfTimer::CompactVisiblePrimitives> gpu_timer(this);
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.compact_visible_primitives");
        executeCompute(
            {{num_splats, 64}},
            &visible_uniforms, sizeof(visible_uniforms),
            pipeline_compact_visible_primitives,
            std::vector<TaggedBinding>{
                {buffers.tiles_touched.deviceBuffer, BufferUse::ComputeRead},
                {buffers.visible_prefix.deviceBuffer, BufferUse::ComputeRead},
                {buffers.primitive_depth_keys.deviceBuffer, BufferUse::ComputeRead},
                {*unsorted_keys, BufferUse::ComputeWrite},
                {*unsorted_idx, BufferUse::ComputeWrite},
            });
    }

    {
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.sort_visible_primitives");
        // Stage 1 sort: num_bits=32 is intentional. Projection writes the full
        // float-as-uint bit pattern of a non-negative radial-distance key, whose
        // unsigned ordering is monotonic. This visible layout contains only the
        // radix command; range construction belongs to the later tile layout.
        // Unsorted keys/idx → radix planned by SortIndirectCountImpl upsweep.
        executeSortIndirectCount(uniforms,
                                 buffers,
                                 32,
                                 buffers.visible_count.deviceBuffer,
                                 buffers.visible_sort_dispatch_args.deviceBuffer,
                                 num_splats,
                                 indirect::VisibleSortDispatch::kLayout,
                                 indirect::VisibleSortDispatch::kRadixWordOffset);
    }

    // Snapshot depth-ranked primitive indices into a stable buffer so stage 2
    // is free to reuse the ping-pong without clobbering the ordering. Matches
    // the CUDA reference's `primitive_indices_sorted` view.
    {
        PerfTimer::Timer<PerfTimer::CopyPrimitiveSortIndices> gpu_timer(this);
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.copy_primitive_sort_indices");
        auto& sort_indices = resizeDeviceBuffer(buffers.primitive_sort_indices, num_splats);
        const DeclaredAccess pre_copy[] = {
            {.buffer = &buffers.sorted_gauss_idx().deviceBuffer, .use = BufferUse::TransferRead},
            {.buffer = &sort_indices, .use = BufferUse::TransferWrite},
        };
        planTransfer(std::span{pre_copy});
        VkBufferCopy copy{};
        copy.srcOffset = buffers.sorted_gauss_idx().deviceBuffer.offset;
        copy.dstOffset = sort_indices.offset;
        copy.size = num_splats * sizeof(int32_t);
        validateBufferRange(buffers.sorted_gauss_idx().deviceBuffer,
                            0,
                            copy.size,
                            "primitive sort-index snapshot source");
        validateBufferRange(sort_indices,
                            0,
                            copy.size,
                            "primitive sort-index snapshot destination");
        if (vulkan_dispatch_.cmd_copy_buffer == nullptr) {
            lfs::rendering::throw_renderer_contract(
                "primitive sort-index snapshot requires VulkanDispatch::cmd_copy_buffer",
                LFS_SOURCE_SITE_CURRENT());
        }
        vulkan_dispatch_.cmd_copy_buffer(command_buffer,
                                         buffers.sorted_gauss_idx().deviceBuffer.buffer,
                                         sort_indices.buffer, 1, &copy);
        const DeclaredAccess post_copy{
            .buffer = &sort_indices,
            .use = BufferUse::ComputeRead,
        };
        planTransfer(std::span{&post_copy, 1});
    }
}

void VulkanGSRenderer::executeApplyDepthOrdering(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers) {
    PerfTimer::Timer<PerfTimer::ApplyDepthOrdering> timer(this);
    DEVICE_GUARD;

    using lfs::rendering::vulkan::BufferUse;

    const size_t num_splats = static_cast<size_t>(uniforms.num_splats);
    if (num_splats == 0)
        return;

    auto& tiles_touched_ordered =
        resizeDeviceBuffer(buffers.tiles_touched_depth_ordered, num_splats);

    struct ApplyUniforms {
        uint32_t num_splats;
        uint32_t pad0, pad1, pad2;
    } apply_uniforms{static_cast<uint32_t>(num_splats), 0, 0, 0};

    executeCompute(
        {{num_splats, 64}},
        &apply_uniforms, sizeof(apply_uniforms),
        pipeline_apply_depth_ordering,
        std::vector<TaggedBinding>{
            {buffers.primitive_sort_indices.deviceBuffer, BufferUse::ComputeRead},
            {buffers.tiles_touched.deviceBuffer, BufferUse::ComputeRead},
            {tiles_touched_ordered, BufferUse::ComputeWrite},
            {buffers.visible_count.deviceBuffer, BufferUse::ComputeRead},
        });
}

void VulkanGSRenderer::executeCullSplats(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    const _VulkanBuffer& transform_indices,
    const _VulkanBuffer& node_mask,
    const _VulkanBuffer& overlay_params,
    const _VulkanBuffer& model_transforms,
    const _VulkanBuffer& lod_indices,
    const _VulkanBuffer& lod_logical_indices,
    const _VulkanBuffer& lod_counts) {
    PerfTimer::Timer<PerfTimer::CullSplats> timer(this);
    DEVICE_GUARD;

    using lfs::rendering::vulkan::BufferUse;

    const size_t num_splats = static_cast<size_t>(uniforms.num_splats);
    if (num_splats == 0)
        return;

    auto& survivors = resizeDeviceBuffer(buffers.survivors, num_splats);
    // clearDeviceBuffer records TransferWrite via planTransfer (§3.2).
    auto& survivor_state = clearDeviceBuffer(
        buffers.survivor_state,
        indirect::SurvivorState::kLayout.word_count);
    validateIndirectLayoutBuffer(survivor_state,
                                 indirect::SurvivorState::kLayout,
                                 "cull_prepare survivor-state producer");
    auto& emit_count = resizeDeviceBuffer(buffers.visible_emit_count, 1);

    // Optional LOD inputs: null → dummy bind to survivor_state (legacy parity).
    // plan() on tagged path covers valid LOD reads (replaces L2592 conditional).
    const _VulkanBuffer lod_indices_binding =
        (lod_indices.buffer != VK_NULL_HANDLE) ? lod_indices : survivor_state;
    const _VulkanBuffer lod_logical_indices_binding =
        (lod_logical_indices.buffer != VK_NULL_HANDLE) ? lod_logical_indices : lod_indices_binding;
    const _VulkanBuffer lod_counts_binding =
        (lod_counts.buffer != VK_NULL_HANDLE) ? lod_counts : survivor_state;

    // Tags from cull_splats.slang CULL_ENTRY_CULL bindings 0–9.
    executeCompute(
        {{num_splats, 256}},
        &uniforms, sizeof(uniforms),
        pipeline_cull_splats,
        std::vector<TaggedBinding>{
            {buffers.xyz_ws.deviceBuffer, BufferUse::ComputeRead},
            {transform_indices, BufferUse::ComputeRead},
            {node_mask, BufferUse::ComputeRead},
            {overlay_params, BufferUse::ComputeRead},
            {model_transforms, BufferUse::ComputeRead},
            {lod_indices_binding, BufferUse::ComputeRead},
            {lod_logical_indices_binding, BufferUse::ComputeRead},
            {lod_counts_binding, BufferUse::ComputeRead},
            {survivors, BufferUse::ComputeWrite},
            {survivor_state, BufferUse::ComputeWrite},
        });

    // Tags from cull_splats.slang CULL_ENTRY_PREPARE: state R/W, emit_count write.
    // L2619 catalog only barriered survivor_state (emit_count mismatch — first write).
    executeCompute(
        {{1, 1}},
        nullptr, 0,
        pipeline_cull_prepare,
        std::vector<TaggedBinding>{
            {survivor_state, BufferUse::ComputeReadWrite},
            {emit_count, BufferUse::ComputeWrite},
        });

    // Post L2629/L2630 deleted: executeProjectionForwardSurvivors (co-migrated)
    // plans survivors/emit_count reads + implicit IndirectRead on survivor_state (§3.4.5).
}

void VulkanGSRenderer::executeProjectionForwardSurvivors(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    const _VulkanBuffer& transform_indices,
    const _VulkanBuffer& node_mask,
    const _VulkanBuffer& overlay_params,
    const _VulkanBuffer& model_transforms,
    size_t visible_capacity,
    const _VulkanBuffer& lod_indices,
    const _VulkanBuffer& lod_logical_indices,
    const _VulkanBuffer& lod_levels,
    const _VulkanBuffer& lod_weights,
    const _VulkanBuffer& lod_counts) {
    PerfTimer::Timer<PerfTimer::ProjectionSurvivors> timer(this);
    DEVICE_GUARD;

    using lfs::rendering::vulkan::BufferUse;

    if (visible_capacity == 0)
        return;

    VulkanGSRendererUniforms survivor_uniforms = uniforms;
    survivor_uniforms.sort_capacity = static_cast<uint32_t>(
        std::min<size_t>(visible_capacity,
                         static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
    if (buffers.quant_pool) {
        survivor_uniforms.lod_page_splats = buffers.pool_page_splats;
    }

    auto& unsorted_keys = resizeDeviceBuffer(buffers.unsorted_keys(), visible_capacity);
    auto& unsorted_idx = resizeDeviceBuffer(buffers.unsorted_gauss_idx(), visible_capacity);
    auto& rect_tile_space = resizeDeviceBuffer(buffers.rect_tile_space, visible_capacity);
    auto& xy_vs = resizeDeviceBuffer(buffers.xy_vs, 2 * visible_capacity);
    auto& depths = resizeDeviceBuffer(buffers.depths, visible_capacity);
    auto& inv_cov = resizeDeviceBuffer(buffers.inv_cov_vs_opacity, 4 * visible_capacity);
    auto& rgb = resizeDeviceBuffer(buffers.rgb, 3 * visible_capacity);
    auto& overlay_flags = resizeDeviceBuffer(buffers.overlay_flags, visible_capacity);
    auto& orig_ids = resizeDeviceBuffer(buffers.orig_ids, visible_capacity);

    const _VulkanBuffer lod_indices_binding =
        (lod_indices.buffer != VK_NULL_HANDLE) ? lod_indices : unsorted_keys;
    const _VulkanBuffer lod_logical_indices_binding =
        (lod_logical_indices.buffer != VK_NULL_HANDLE) ? lod_logical_indices : lod_indices_binding;
    const _VulkanBuffer lod_levels_binding =
        (lod_levels.buffer != VK_NULL_HANDLE) ? lod_levels : unsorted_keys;
    const _VulkanBuffer lod_weights_binding =
        (lod_weights.buffer != VK_NULL_HANDLE) ? lod_weights : unsorted_keys;
    const _VulkanBuffer lod_counts_binding =
        (lod_counts.buffer != VK_NULL_HANDLE) ? lod_counts : unsorted_keys;

    // Dense buffer array indexed by binding number (catalog appendix L2735).
    // Pipeline layouts skip bindings 6 and 8 (placeholders still occupy slots).
    // Tags: attr/LOD/survivors/state reads; compact outputs write; emit_count R/W atomic.
    std::vector<TaggedBinding> tagged = {
        {buffers.xyz_ws.deviceBuffer, BufferUse::ComputeRead},                  // 0
        {buffers.sh0.deviceBuffer, BufferUse::ComputeRead},                     // 1
        {buffers.shN.deviceBuffer, BufferUse::ComputeRead},                     // 2
        {buffers.rotations.deviceBuffer, BufferUse::ComputeRead},               // 3
        {buffers.scaling_raw.deviceBuffer, BufferUse::ComputeRead},             // 4
        {buffers.opacity_raw.deviceBuffer, BufferUse::ComputeRead},             // 5
        {unsorted_keys, BufferUse::ComputeWrite},                               // 6 placeholder
        {rect_tile_space, BufferUse::ComputeWrite},                             // 7
        {unsorted_keys, BufferUse::ComputeWrite},                               // 8 placeholder
        {xy_vs, BufferUse::ComputeWrite},                                       // 9
        {depths, BufferUse::ComputeWrite},                                      // 10
        {inv_cov, BufferUse::ComputeWrite},                                     // 11
        {rgb, BufferUse::ComputeWrite},                                         // 12
        {overlay_flags, BufferUse::ComputeWrite},                               // 13
        {transform_indices, BufferUse::ComputeRead},                            // 14
        {node_mask, BufferUse::ComputeRead},                                    // 15
        {overlay_params, BufferUse::ComputeRead},                               // 16
        {model_transforms, BufferUse::ComputeRead},                             // 17
        {unsorted_keys, BufferUse::ComputeWrite},                               // 18 placeholder
        {lod_indices_binding, BufferUse::ComputeRead},                          // 19
        {lod_logical_indices_binding, BufferUse::ComputeRead},                  // 20
        {lod_levels_binding, BufferUse::ComputeRead},                           // 21
        {lod_weights_binding, BufferUse::ComputeRead},                          // 22
        {lod_counts_binding, BufferUse::ComputeRead},                           // 23
        {buffers.survivors.deviceBuffer, BufferUse::ComputeRead},               // 24
        {buffers.survivor_state.deviceBuffer, BufferUse::ComputeRead},          // 25
        {unsorted_idx, BufferUse::ComputeWrite},                                // 26
        {buffers.visible_emit_count.deviceBuffer, BufferUse::ComputeReadWrite}, // 27
        {orig_ids, BufferUse::ComputeWrite},                                    // 28
    };
    if (buffers.quant_pool) {
        tagged.push_back({buffers.page_frames.deviceBuffer, BufferUse::ComputeRead}); // 29
    }

    // Indirect: plan() adds implicit IndirectRead on survivor_state (replaces L2629 handoff).
    executeComputeIndirect(
        buffers.survivor_state.deviceBuffer,
        indirect::byteOffset(indirect::SurvivorState::kProjectionWordOffset),
        &survivor_uniforms, sizeof(survivor_uniforms),
        buffers.quant_pool ? pipeline_projection_forward_quant_survivors
                           : pipeline_projection_forward_survivors,
        tagged);
}

void VulkanGSRenderer::executeSortPrimitivesByDepthVisible(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    size_t visible_capacity) {
    PerfTimer::Timer<PerfTimer::SortPrimitivesByDepth> timer(this);
    DEVICE_GUARD;

    if (visible_capacity == 0)
        return;

    resizeDeviceBuffer(buffers.visible_count, 2);
    auto& visible_dispatch = resizeDeviceBuffer(
        buffers.visible_dispatch,
        indirect::VisibleChainDispatch::kLayout.word_count);
    validateIndirectLayoutBuffer(visible_dispatch,
                                 indirect::VisibleChainDispatch::kLayout,
                                 "prepare_visible_chain producer");
    auto& cumsum_counts = resizeDeviceBuffer(buffers.cumsum_counts, 4);

    struct PrepareUniforms {
        uint32_t visible_capacity;
        uint32_t sort_partition_size;
        uint32_t pad0, pad1;
    } prepare_uniforms{
        static_cast<uint32_t>(
            std::min<size_t>(visible_capacity,
                             static_cast<size_t>(std::numeric_limits<uint32_t>::max()))),
        512u * 8u, 0, 0};

    using lfs::rendering::vulkan::BufferUse;
    using lfs::rendering::vulkan::DeclaredAccess;

    {
        PerfTimer::Timer<PerfTimer::PrepareVisibleSort> gpu_timer(this);
        // Tags from prepare_visible_chain.slang.
        executeCompute(
            {{1, 1}},
            &prepare_uniforms, sizeof(prepare_uniforms),
            pipeline_prepare_visible_chain,
            std::vector<TaggedBinding>{
                {buffers.visible_emit_count.deviceBuffer, BufferUse::ComputeRead},
                {buffers.visible_count.deviceBuffer, BufferUse::ComputeWrite},
                {visible_dispatch, BufferUse::ComputeWrite},
                {cumsum_counts, BufferUse::ComputeWrite},
            });
        // Handoff: count/cumsum for transfer+compute consumers; dispatch for
        // IndirectRead is planned by subsequent executeComputeIndirect.
        const DeclaredAccess handoff[] = {
            {.buffer = &buffers.visible_count.deviceBuffer, .use = BufferUse::TransferRead},
            {.buffer = &buffers.visible_count.deviceBuffer, .use = BufferUse::ComputeRead},
            {.buffer = &cumsum_counts, .use = BufferUse::TransferRead},
            {.buffer = &cumsum_counts, .use = BufferUse::ComputeRead},
        };
        planTransfer(std::span{handoff});
        // The readback's bound must be the render domain: clamping the raw
        // count to the frame's *capacity* would mask exactly the clamping the
        // raw count exists to detect.
        recordVisibleCountReadback(buffers, static_cast<size_t>(uniforms.num_splats));
    }

    {
        // Unsorted keys → radix planned by SortIndirectCountImpl.
        executeSortIndirectCount(uniforms,
                                 buffers,
                                 32,
                                 buffers.visible_count.deviceBuffer,
                                 visible_dispatch,
                                 visible_capacity,
                                 indirect::VisibleChainDispatch::kLayout,
                                 indirect::VisibleChainDispatch::kRadixWordOffset);
    }

    {
        PerfTimer::Timer<PerfTimer::CopyPrimitiveSortIndices> gpu_timer(this);
        auto& sort_indices = resizeDeviceBuffer(buffers.primitive_sort_indices, visible_capacity);
        struct CopyUniforms {
            uint32_t capacity;
            uint32_t pad0, pad1, pad2;
        } copy_uniforms{prepare_uniforms.visible_capacity, 0, 0, 0};
        executeComputeIndirect(
            visible_dispatch,
            indirect::byteOffset(indirect::VisibleChainDispatch::kPerElementWordOffset),
            &copy_uniforms, sizeof(copy_uniforms),
            pipeline_copy_visible_indices,
            std::vector<TaggedBinding>{
                {buffers.sorted_gauss_idx().deviceBuffer, BufferUse::ComputeRead},
                {sort_indices, BufferUse::ComputeWrite},
                {buffers.visible_count.deviceBuffer, BufferUse::ComputeRead},
            });
        // Handoff for downstream compute consumers of sort_indices.
        const DeclaredAccess post{
            .buffer = &sort_indices,
            .use = BufferUse::ComputeRead,
        };
        planTransfer(std::span{&post, 1});
    }
}

void VulkanGSRenderer::executeMacroCoverage(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    size_t visible_capacity) {
    PerfTimer::Timer<PerfTimer::ApplyDepthOrdering> timer(this);
    DEVICE_GUARD;

    using lfs::rendering::vulkan::BufferUse;

    if (visible_capacity == 0)
        return;

    // Reuses tiles_touched_depth_ordered as the per-rank macro coverage
    // counts; the visible-bounded cumsum into index_buffer_offset is shared
    // with the render-tile chain.
    auto& macro_counts =
        resizeDeviceBuffer(buffers.tiles_touched_depth_ordered, visible_capacity);

    executeComputeIndirect(
        buffers.visible_dispatch.deviceBuffer,
        indirect::byteOffset(indirect::VisibleChainDispatch::kPerElementWordOffset),
        &uniforms, sizeof(uniforms),
        pipeline_macro_coverage,
        std::vector<TaggedBinding>{
            {buffers.primitive_sort_indices.deviceBuffer, BufferUse::ComputeRead},
            {buffers.rect_tile_space.deviceBuffer, BufferUse::ComputeRead},
            {macro_counts, BufferUse::ComputeWrite},
            {buffers.visible_count.deviceBuffer, BufferUse::ComputeRead},
            {buffers.xy_vs.deviceBuffer, BufferUse::ComputeRead},
            {buffers.inv_cov_vs_opacity.deviceBuffer, BufferUse::ComputeRead},
        });
}

void VulkanGSRenderer::executeMacroDepthWaves(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    const size_t armed,
    const int sort_bits,
    const _VulkanBuffer& selection_mask,
    const _VulkanBuffer& preview_mask,
    const _VulkanBuffer& selection_colors,
    const _VulkanBuffer& overlay_params,
    const bool overlays_active,
    const bool predicate_waves) {
    PerfTimer::Timer<PerfTimer::RasterizeForward> timer(this);
    DEVICE_GUARD;

    if (armed == 0 || uniforms.sort_capacity != HIGS_DEPTH_WAVE_INSTANCES ||
        sort_bits <= 0 || sort_bits > 32) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Macro depth waves require non-zero slots, fixed K, and sort bits in [1,32] (armed={}, uniform_capacity={}, K={}, sort_bits={})",
                armed,
                uniforms.sort_capacity,
                HIGS_DEPTH_WAVE_INSTANCES,
                sort_bits),
            LFS_SOURCE_SITE_CURRENT());
    }
    const auto wave_layout = indirect::DepthWave::layout(armed);
    validateIndirectLayoutBuffer(buffers.depth_wave_dispatch.deviceBuffer,
                                 wave_layout,
                                 "macro depth-wave consumer");
    if (buffers.wave_predicates.deviceBuffer.size < armed * sizeof(uint32_t)) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Macro depth waves require one predicate per slot (armed={}, predicate_bytes={}, required_bytes={})",
                armed,
                buffers.wave_predicates.deviceBuffer.size,
                armed * sizeof(uint32_t)),
            LFS_SOURCE_SITE_CURRENT());
    }

    const size_t capacity = HIGS_DEPTH_WAVE_INSTANCES;
    // Logical macro-grid and pixel counts (dispatch/uniforms/indexing).
    const size_t num_macro =
        _CEIL_DIV(static_cast<size_t>(uniforms.grid_width), size_t{HIGS_MACRO_T16_W}) *
        _CEIL_DIV(static_cast<size_t>(uniforms.grid_height), size_t{HIGS_MACRO_T16_H});
    const size_t num_pixels =
        static_cast<size_t>(uniforms.image_height) * uniforms.image_width;
    if (num_macro == 0 || num_pixels == 0)
        return;

    // Capacity for pixel/tile-derived scratch: 64-px bucket (issue #1565).
    const auto scratch_bucket = lfs::rendering::vulkan::viewportScratchBucket(
        uniforms.image_width, uniforms.image_height);
    const size_t alloc_pixels = scratch_bucket.alloc_pixels;
    const size_t alloc_tiles = scratch_bucket.alloc_tiles;
    const size_t alloc_grid_w =
        _CEIL_DIV(static_cast<size_t>(scratch_bucket.alloc_w), size_t{TILE_WIDTH});
    const size_t alloc_grid_h =
        _CEIL_DIV(static_cast<size_t>(scratch_bucket.alloc_h), size_t{TILE_HEIGHT});
    const size_t alloc_macro_tiles =
        _CEIL_DIV(alloc_grid_w, size_t{HIGS_MACRO_T16_W}) *
        _CEIL_DIV(alloc_grid_h, size_t{HIGS_MACRO_T16_H});
    if (scratch_bucket.alloc_w != scratch_bucket_alloc_w_ ||
        scratch_bucket.alloc_h != scratch_bucket_alloc_h_) {
        LOG_DEBUG(
            "vksplat.scratch.bucket logical={}x{} alloc={}x{} pixels_cap={} tiles_cap={}",
            uniforms.image_width,
            uniforms.image_height,
            scratch_bucket.alloc_w,
            scratch_bucket.alloc_h,
            alloc_pixels,
            alloc_tiles);
        scratch_bucket_alloc_w_ = scratch_bucket.alloc_w;
        scratch_bucket_alloc_h_ = scratch_bucket.alloc_h;
    }
    const auto resize_scratch = [this](auto& typed_buffer, const size_t elements) -> _VulkanBuffer& {
        auto& dev = typed_buffer.deviceBuffer;
        const size_t old_capacity = dev.capacity;
        auto& result = resizeDeviceBuffer(typed_buffer, elements);
        if (result.capacity != old_capacity) {
            LOG_DEBUG(
                "vksplat.scratch.realloc label={} old_bytes={} new_bytes={}",
                result.label != nullptr ? result.label : "<unlabeled>",
                old_capacity,
                result.capacity);
        }
        return result;
    };

    // Wave budget stays on the logical macro grid so padded capacity cannot
    // false-trip HIGS_RASTER_MAX_WAVES. Buffer capacities use the bucketed grid.
    const size_t max_batches =
        _CEIL_DIV(capacity, size_t{RASTER_BATCH_SIZE}) + num_macro;
    const size_t alloc_max_batches =
        _CEIL_DIV(capacity, size_t{RASTER_BATCH_SIZE}) + alloc_macro_tiles;
    const size_t batch_waves =
        _CEIL_DIV(max_batches, size_t{HIGS_RASTER_WAVE_BATCHES});
    if (batch_waves > HIGS_RASTER_MAX_WAVES) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Macro raster batch-wave budget exceeded: fixed K and grid require {} waves of {} armed (K={}, num_macro={}, max_batches={})",
                batch_waves,
                HIGS_RASTER_MAX_WAVES,
                capacity,
                num_macro,
                max_batches),
            LFS_SOURCE_SITE_CURRENT());
    }
    const size_t pool_batches =
        std::min<size_t>(max_batches, HIGS_RASTER_WAVE_BATCHES);
    const size_t alloc_pool_batches =
        std::min<size_t>(alloc_max_batches, HIGS_RASTER_WAVE_BATCHES);

    // Every allocation below is content-independent: K, viewport geometry, or
    // a fixed indirect-layout size. Do this before opening conditional blocks.
    resizeDeviceBuffer(buffers.sorting_keys_1, capacity);
    resizeDeviceBuffer(buffers.sorting_keys_2, capacity);
    resizeDeviceBuffer(buffers.sorting_gauss_idx_1, capacity);
    resizeDeviceBuffer(buffers.sorting_gauss_idx_2, capacity);
    auto& tile_ranges = resize_scratch(buffers.tile_ranges, alloc_macro_tiles + 1u);
    auto& batch_counts = resize_scratch(buffers.tile_batch_counts, alloc_macro_tiles);
    auto& batch_offsets = resize_scratch(buffers.tile_batch_offsets, alloc_macro_tiles);
    auto& macro_wave_args = resizeDeviceBuffer(
        buffers.macro_wave_args,
        indirect::MacroWaveDispatch::kLayout.word_count);
    validateIndirectLayoutBuffer(macro_wave_args,
                                 indirect::MacroWaveDispatch::kLayout,
                                 "macro_batch_prepare producer");
    auto& partials = resizeDeviceBuffer(
        buffers.macro_partials,
        alloc_pool_batches * HIGS_MACRO_TILE_SIZE_TILES * HIGS_TILE_SIZE * 4u);
    auto& active_mask = resize_scratch(buffers.macro_active_mask, alloc_max_batches);
    auto& pixel_state = resize_scratch(buffers.pixel_state, 4u * alloc_pixels);
    auto& pixel_depth = resize_scratch(buffers.pixel_depth, alloc_pixels);
    auto& n_contributors = resize_scratch(buffers.n_contributors, alloc_pixels);

    constexpr size_t kRadix = 256u;
    constexpr size_t kPartitionSize = 512u * 8u;
    const size_t radix_passes = _CEIL_DIV(static_cast<size_t>(sort_bits), size_t{8});
    resizeDeviceBuffer(buffers._sorting_histogram, radix_passes * kRadix);
    resizeDeviceBuffer(buffers._sorting_histogram_cumsum,
                       _CEIL_DIV(capacity, kPartitionSize) * kRadix);
    constexpr size_t kCumsumBlock = 1024u;
    // Cumsum capacity follows the bucketed macro-tile grid so resize-drag steps
    // inside a 64-px bucket do not reallocate (logical work still uses num_macro).
    resizeDeviceBuffer(buffers._cumsum_blockSums,
                       std::max<size_t>(1u, _CEIL_DIV(alloc_macro_tiles, kCumsumBlock)));
    resizeDeviceBuffer(
        buffers._cumsum_blockSums2,
        std::max<size_t>(1u,
                         _CEIL_DIV(_CEIL_DIV(alloc_macro_tiles, kCumsumBlock),
                                   kCumsumBlock)));

    const bool use_fp32 = overlays_active || (uniforms.lod_enabled & 4u) != 0u;
    auto& raster_pipeline = overlays_active
                                ? pipeline_macro_raster_overlays
                                : (use_fp32 ? pipeline_macro_raster_fp32
                                            : pipeline_macro_raster);
    auto& compose_pipeline = overlays_active
                                 ? pipeline_macro_compose_overlays
                                 : pipeline_macro_compose;

    using lfs::rendering::vulkan::BufferUse;
    using lfs::rendering::vulkan::DeclaredAccess;
    // Mega-hoist removed: per-wave tagged dispatches plan hazards exactly.

    const auto& wave_buffer = buffers.depth_wave_dispatch.deviceBuffer;
    const auto& predicate_buffer = buffers.wave_predicates.deviceBuffer;
    for (size_t wave = 0; wave < armed; ++wave) {
        const bool conditional = predicate_waves && supports_conditional_rendering_;
        if (conditional) {
            const DeclaredAccess pred{
                .buffer = &predicate_buffer,
                .use = BufferUse::ConditionalRead,
            };
            planTransfer(std::span{&pred, 1});
        }
        const ConditionalRenderingScope conditional_scope(
            *this,
            conditional,
            vk_cmd_begin_conditional_rendering_,
            vk_cmd_end_conditional_rendering_,
            predicate_buffer,
            wave * sizeof(uint32_t));

        VulkanGSRendererUniforms wave_uniforms = uniforms;
        wave_uniforms.depth_wave = static_cast<uint32_t>(wave);
        const auto record = bufferView(
            wave_buffer,
            indirect::byteOffset(indirect::DepthWave::recordWordOffset(wave)),
            indirect::byteSize(indirect::DepthWave::kRecordLayout));
        const auto count = bufferView(
            wave_buffer,
            indirect::byteOffset(indirect::DepthWave::countWordOffset(wave)),
            2u * sizeof(uint32_t));

        auto& unsorted_keys = buffers.unsorted_keys().deviceBuffer;
        auto& unsorted_indices = buffers.unsorted_gauss_idx().deviceBuffer;
        // pipeline_generate_macro_keys_wave: 9 bindings.
        executeComputeIndirect(
            record,
            indirect::byteOffset(indirect::DepthWave::kKeygenWordOffset),
            &wave_uniforms,
            sizeof(wave_uniforms),
            pipeline_generate_macro_keys_wave,
            std::vector<TaggedBinding>{
                {buffers.xy_vs.deviceBuffer, BufferUse::ComputeRead},
                {buffers.inv_cov_vs_opacity.deviceBuffer, BufferUse::ComputeRead},
                {buffers.rect_tile_space.deviceBuffer, BufferUse::ComputeRead},
                {buffers.index_buffer_offset.deviceBuffer, BufferUse::ComputeRead},
                {buffers.primitive_sort_indices.deviceBuffer, BufferUse::ComputeRead},
                {unsorted_keys, BufferUse::ComputeWrite},
                {unsorted_indices, BufferUse::ComputeWrite},
                {buffers.visible_count.deviceBuffer, BufferUse::ComputeRead},
                {wave_buffer, BufferUse::ComputeRead},
            });

        executeSortIndirectCountImpl(wave_uniforms,
                                     buffers,
                                     sort_bits,
                                     count,
                                     record,
                                     capacity,
                                     indirect::DepthWave::kRecordLayout,
                                     indirect::DepthWave::kRadixWordOffset,
                                     "vksplat.render.record.sort_macro_depth_wave",
                                     true);

        executeComputeIndirect(
            record,
            indirect::byteOffset(indirect::DepthWave::kPerTileWordOffset),
            &wave_uniforms,
            sizeof(wave_uniforms),
            pipeline_compute_macro_ranges[buffers.is_unsorted_1],
            std::vector<TaggedBinding>{
                {buffers.sorted_keys().deviceBuffer, BufferUse::ComputeRead},
                {tile_ranges, BufferUse::ComputeWrite},
                {count, BufferUse::ComputeRead},
                {batch_counts, BufferUse::ComputeWrite},
            });

        executeCumsum(buffers,
                      buffers.tile_batch_counts,
                      buffers.tile_batch_offsets,
                      wave < HIGS_DEPTH_MAX_WAVES);
        executeCompute({{1, 1}},
                       &wave_uniforms,
                       sizeof(wave_uniforms),
                       pipeline_macro_batch_prepare,
                       std::vector<TaggedBinding>{
                           {batch_offsets, BufferUse::ComputeRead},
                           {macro_wave_args, BufferUse::ComputeWrite},
                       });

        std::vector<TaggedBinding> raster_bindings{
            {buffers.sorted_gauss_idx().deviceBuffer, BufferUse::ComputeRead},
            {tile_ranges, BufferUse::ComputeRead},
            {batch_offsets, BufferUse::ComputeRead},
            {buffers.xy_vs.deviceBuffer, BufferUse::ComputeRead},
            {buffers.inv_cov_vs_opacity.deviceBuffer, BufferUse::ComputeRead},
            {buffers.rgb.deviceBuffer, BufferUse::ComputeRead},
            {partials, BufferUse::ComputeReadWrite},
            {active_mask, BufferUse::ComputeReadWrite},
        };
        if (overlays_active) {
            raster_bindings.insert(raster_bindings.end(),
                                   {{selection_mask, BufferUse::ComputeRead},
                                    {preview_mask, BufferUse::ComputeRead},
                                    {selection_colors, BufferUse::ComputeRead},
                                    {buffers.overlay_flags.deviceBuffer, BufferUse::ComputeRead},
                                    {overlay_params, BufferUse::ComputeRead},
                                    {buffers.orig_ids.deviceBuffer, BufferUse::ComputeRead}});
        }
        std::vector<TaggedBinding> compose_bindings{
            {buffers.sorted_gauss_idx().deviceBuffer, BufferUse::ComputeRead},
            {tile_ranges, BufferUse::ComputeRead},
            {batch_offsets, BufferUse::ComputeRead},
            {buffers.xy_vs.deviceBuffer, BufferUse::ComputeRead},
            {buffers.inv_cov_vs_opacity.deviceBuffer, BufferUse::ComputeRead},
            {buffers.rgb.deviceBuffer, BufferUse::ComputeRead},
            {buffers.depths.deviceBuffer, BufferUse::ComputeRead},
            {partials, BufferUse::ComputeRead},
            {active_mask, BufferUse::ComputeRead},
            {pixel_state, BufferUse::ComputeReadWrite},
            {pixel_depth, BufferUse::ComputeReadWrite},
            {n_contributors, BufferUse::ComputeReadWrite},
        };
        if (overlays_active) {
            compose_bindings.insert(compose_bindings.end(),
                                    {{selection_mask, BufferUse::ComputeRead},
                                     {preview_mask, BufferUse::ComputeRead},
                                     {selection_colors, BufferUse::ComputeRead},
                                     {buffers.overlay_flags.deviceBuffer, BufferUse::ComputeRead},
                                     {overlay_params, BufferUse::ComputeRead},
                                     {buffers.orig_ids.deviceBuffer, BufferUse::ComputeRead}});
        }

        for (size_t batch_wave = 0; batch_wave < batch_waves; ++batch_wave) {
            wave_uniforms.wave_base =
                static_cast<uint32_t>(batch_wave * HIGS_RASTER_WAVE_BATCHES);
            executeComputeIndirect(
                macro_wave_args,
                indirect::byteOffset(indirect::MacroWaveDispatch::rasterWordOffset(batch_wave)),
                &wave_uniforms,
                sizeof(wave_uniforms),
                raster_pipeline[buffers.is_unsorted_1],
                raster_bindings);
            executeComputeIndirect(
                macro_wave_args,
                indirect::byteOffset(indirect::MacroWaveDispatch::composeWordOffset(batch_wave)),
                &wave_uniforms,
                sizeof(wave_uniforms),
                compose_pipeline[buffers.is_unsorted_1],
                compose_bindings);
        }
    }
}

void VulkanGSRenderer::executeCalculateIndexBufferOffsetVisible(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    size_t visible_capacity) {
    PerfTimer::Timer<PerfTimer::CalculateIndexBufferOffset> timer(this);
    DEVICE_GUARD;

    if (visible_capacity == 0) {
        buffers.num_indices = 0;
        return;
    }

    const size_t block = 1024;
    const size_t c1_capacity = _CEIL_DIV(visible_capacity, block);
    const size_t c2_capacity = _CEIL_DIV(c1_capacity, block);
    if (c2_capacity > block) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Visible capacity exceeds the three-level indirect cumsum range (visible_capacity={}, block_size={}, level1_capacity={}, level2_capacity={}, max_level2_capacity={})",
                visible_capacity,
                block,
                c1_capacity,
                c2_capacity,
                block),
            LFS_SOURCE_SITE_CURRENT());
    }

    using lfs::rendering::vulkan::BufferUse;
    using lfs::rendering::vulkan::DeclaredAccess;

    auto& input = buffers.tiles_touched_depth_ordered.deviceBuffer;
    auto& output = resizeDeviceBuffer(buffers.index_buffer_offset, visible_capacity);
    auto& block_sums = resizeDeviceBuffer(buffers._cumsum_blockSums, c1_capacity, true);
    auto& block_sums2 = resizeDeviceBuffer(buffers._cumsum_blockSums2, c2_capacity, true);
    auto& counts = buffers.cumsum_counts.deviceBuffer;
    auto& dispatch = buffers.visible_dispatch.deviceBuffer;

    // Same host contract as classic executeCumsum (sweep_c C2.1): GPU indexes
    // g_input[gid] for gid < visible-derived element counts without a length API.
    const auto require_backing = [](const _VulkanBuffer& b, const size_t needed_elements,
                                    const char* role) {
        const size_t needed_bytes = needed_elements * sizeof(int32_t);
        if (b.buffer == VK_NULL_HANDLE || b.allocSize < needed_bytes || b.size < needed_bytes) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "VkSplat visible-chain cumsum {} backing is smaller than the scan (label='{}', buffer={:#x}, alloc_bytes={}, capacity_bytes={}, active_bytes={}, needed_bytes={}, elements={})",
                    role,
                    b.label ? b.label : "?",
                    lfs::rendering::vkHandleValue(b.buffer),
                    b.allocSize,
                    b.capacity,
                    b.size,
                    needed_bytes,
                    needed_elements),
                LFS_SOURCE_SITE_CURRENT());
        }
    };
    require_backing(input, visible_capacity, "input");
    require_backing(output, visible_capacity, "output");
    require_backing(block_sums, c1_capacity, "block_sums");
    require_backing(block_sums2, c2_capacity, "block_sums2");

    const auto level_uniform = [](uint32_t level) { return level; };

    // Indirect cumsum tags (cumsum.slang CUMSUM_INDIRECT): input R, output W,
    // blockSums RW, counts R. plan() merges per buffer; inter-phase barriers
    // are derived. Implicit IndirectRead on dispatch for each indirect phase.
    {
        PerfTimer::Timer<PerfTimer::_Cumsum> cumsum_timer(this);

        // Always-recorded 3-level indirect scan. Degenerate levels dispatch a
        // single group over 1 element, so no host-side branching on the count.
        uint32_t level = level_uniform(0);
        executeComputeIndirect(dispatch,
                               indirect::byteOffset(indirect::VisibleChainDispatch::kCumsumLevel0WordOffset),
                               &level, sizeof(level),
                               pipeline_cumsum_indirect.block_scan,
                               std::vector<TaggedBinding>{
                                   {input, BufferUse::ComputeRead},
                                   {output, BufferUse::ComputeWrite},
                                   {block_sums, BufferUse::ComputeReadWrite},
                                   {counts, BufferUse::ComputeRead},
                               });

        level = level_uniform(1);
        executeComputeIndirect(dispatch,
                               indirect::byteOffset(indirect::VisibleChainDispatch::kCumsumLevel1WordOffset),
                               &level, sizeof(level),
                               pipeline_cumsum_indirect.block_scan,
                               std::vector<TaggedBinding>{
                                   {block_sums, BufferUse::ComputeRead},
                                   {block_sums, BufferUse::ComputeWrite},
                                   {block_sums2, BufferUse::ComputeReadWrite},
                                   {counts, BufferUse::ComputeRead},
                               });

        level = level_uniform(2);
        executeCompute({{1, 1}},
                       &level, sizeof(level),
                       pipeline_cumsum_indirect.scan_block_sums,
                       std::vector<TaggedBinding>{
                           {block_sums, BufferUse::ComputeRead},
                           {block_sums, BufferUse::ComputeWrite},
                           {block_sums2, BufferUse::ComputeReadWrite},
                           {counts, BufferUse::ComputeRead},
                       });

        level = level_uniform(1);
        executeComputeIndirect(dispatch,
                               indirect::byteOffset(indirect::VisibleChainDispatch::kCumsumLevel1WordOffset),
                               &level, sizeof(level),
                               pipeline_cumsum_indirect.add_block_offsets,
                               std::vector<TaggedBinding>{
                                   {block_sums, BufferUse::ComputeRead},
                                   {block_sums, BufferUse::ComputeWrite},
                                   {block_sums2, BufferUse::ComputeReadWrite},
                                   {counts, BufferUse::ComputeRead},
                               });

        level = level_uniform(0);
        executeComputeIndirect(dispatch,
                               indirect::byteOffset(indirect::VisibleChainDispatch::kCumsumLevel0WordOffset),
                               &level, sizeof(level),
                               pipeline_cumsum_indirect.add_block_offsets,
                               std::vector<TaggedBinding>{
                                   {input, BufferUse::ComputeRead},
                                   {output, BufferUse::ComputeWrite},
                                   {block_sums, BufferUse::ComputeReadWrite},
                                   {counts, BufferUse::ComputeRead},
                               });
    }

    {
        PerfTimer::Timer<PerfTimer::PrepareTileSort> gpu_timer(this);
        resizeDeviceBuffer(buffers.tile_sort_count, 1);
        if (buffers.tile_sort_count.deviceBuffer.size != sizeof(uint32_t)) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "Visible-chain prepare_tile_sort count buffer must contain exactly one uint32 word (buffer={:#x}, active_bytes={}, allocation_bytes={}, required_bytes={})",
                    lfs::rendering::vkHandleValue(buffers.tile_sort_count.deviceBuffer.buffer),
                    buffers.tile_sort_count.deviceBuffer.size,
                    buffers.tile_sort_count.deviceBuffer.allocSize,
                    sizeof(uint32_t)),
                LFS_SOURCE_SITE_CURRENT());
        }
        const uint32_t visible_limit = static_cast<uint32_t>(
            std::min<size_t>(visible_capacity,
                             static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
        // Visible path indexes index_buffer_offset[tail-1] with tail ≤ visible_limit.
        if (visible_limit > 0 &&
            buffers.index_buffer_offset.deviceSize() < static_cast<size_t>(visible_limit)) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                    "prepare_tile_sort_visible requires index_buffer_offset covering visible_limit (visible_limit={}, device_elements={}, buffer={:#x})",
                    visible_limit,
                    buffers.index_buffer_offset.deviceSize(),
                    lfs::rendering::vkHandleValue(buffers.index_buffer_offset.deviceBuffer.buffer)),
                LFS_SOURCE_SITE_CURRENT());
        }

        // prepare_tile_sort.slang with visible_count binding.
        executeCompute(
            {{1, 1}},
            &visible_limit, sizeof(visible_limit),
            pipeline_prepare_tile_sort_visible,
            std::vector<TaggedBinding>{
                {output, BufferUse::ComputeRead},
                {buffers.tile_sort_count.deviceBuffer, BufferUse::ComputeWrite},
                {buffers.visible_count.deviceBuffer, BufferUse::ComputeRead},
            });
    }

    const DeclaredAccess tile_count_handoff[] = {
        {.buffer = &buffers.tile_sort_count.deviceBuffer, .use = BufferUse::TransferRead},
        {.buffer = &buffers.tile_sort_count.deviceBuffer, .use = BufferUse::ComputeRead},
    };
    planTransfer(std::span{tile_count_handoff});
}

void VulkanGSRenderer::executeWavePartition(const VulkanGSRendererUniforms& uniforms,
                                            VulkanGSPipelineBuffers& buffers,
                                            const size_t armed,
                                            const bool visible_bounded) {
    DEVICE_GUARD;
    if (armed == 0 || armed > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        lfs::rendering::throw_renderer_contract(
            std::format("Depth-wave partition requires a non-zero uint32 slot count (armed={})",
                        armed),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (uniforms.sort_capacity != HIGS_DEPTH_WAVE_INSTANCES) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Depth-wave partition requires the fixed K budget (uniform_capacity={}, K={}, armed={}, visible_bounded={})",
                uniforms.sort_capacity,
                HIGS_DEPTH_WAVE_INSTANCES,
                armed,
                visible_bounded),
            LFS_SOURCE_SITE_CURRENT());
    }

    const size_t num_tiles = visible_bounded
                                 ? static_cast<size_t>(_CEIL_DIV(uniforms.grid_width,
                                                                 HIGS_MACRO_T16_W)) *
                                       _CEIL_DIV(uniforms.grid_height, HIGS_MACRO_T16_H)
                                 : static_cast<size_t>(uniforms.grid_width) *
                                       uniforms.grid_height;
    if (num_tiles > HIGS_DEPTH_WAVE_INSTANCES / 2u) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "Depth-wave K floor violated: K={} must be at least twice max_rank_emission={} (chain={})",
                HIGS_DEPTH_WAVE_INSTANCES,
                num_tiles,
                visible_bounded ? "macro" : "legacy"),
            LFS_SOURCE_SITE_CURRENT());
    }

    auto& wave_dispatch = resizeDeviceBuffer(
        buffers.depth_wave_dispatch,
        indirect::DepthWave::layout(armed).word_count);
    buffers.wave_predicates.deviceBuffer.extra_usage =
        supports_conditional_rendering_ ? VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT : 0u;
    auto& predicates = resizeDeviceBuffer(buffers.wave_predicates, armed);
    validateIndirectLayoutBuffer(wave_dispatch,
                                 indirect::DepthWave::layout(armed),
                                 "depth-wave partition producer");

    using lfs::rendering::vulkan::BufferUse;
    using lfs::rendering::vulkan::DeclaredAccess;

    VulkanGSRendererUniforms partition_uniforms = uniforms;
    partition_uniforms.sort_capacity = HIGS_DEPTH_WAVE_INSTANCES;
    partition_uniforms.depth_wave = static_cast<uint32_t>(armed);

    // pipeline_wave_partition: 4 bindings; visible variant: 5 (inserts visible_count).
    std::vector<TaggedBinding> bindings{
        {buffers.index_buffer_offset.deviceBuffer, BufferUse::ComputeRead},
    };
    if (visible_bounded) {
        bindings.push_back({buffers.visible_count.deviceBuffer, BufferUse::ComputeRead});
    }
    bindings.push_back({buffers.tile_sort_count.deviceBuffer, BufferUse::ComputeRead});
    bindings.push_back({wave_dispatch, BufferUse::ComputeWrite});
    bindings.push_back({predicates, BufferUse::ComputeWrite});
    executeCompute({{1, 1}},
                   &partition_uniforms,
                   sizeof(partition_uniforms),
                   visible_bounded ? pipeline_wave_partition_visible
                                   : pipeline_wave_partition,
                   bindings);

    // Handoffs: waves consume dispatch as IndirectRead (planned by per-wave
    // executeComputeIndirect); tile_sort_count for transfer/host; predicates for
    // ConditionalRenderingScope which has no barrier site of its own (§3.4.5).
    std::vector<DeclaredAccess> handoff{
        {.buffer = &wave_dispatch, .use = BufferUse::IndirectRead},
        {.buffer = &buffers.tile_sort_count.deviceBuffer, .use = BufferUse::TransferRead},
        {.buffer = &buffers.tile_sort_count.deviceBuffer, .use = BufferUse::ComputeRead},
    };
    if (supports_conditional_rendering_) {
        handoff.push_back({.buffer = &predicates, .use = BufferUse::ConditionalRead});
    }
    planTransfer(std::span{handoff});
    recordInstanceCountReadback(buffers, armed);
}
