#include "gs_renderer.h"

#include "core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <csignal>
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

    VkBufferCopy copy{};
    copy.srcOffset = buffers.tile_sort_count.deviceBuffer.offset;
    copy.dstOffset = 0;
    copy.size = sizeof(uint32_t);
    validateBufferRange(count_buffer, 0, copy.size, "tile-instance count readback source");
    validateBufferRange(instance_count_readback_buffer_, 0, copy.size, "tile-instance count readback destination");
    vkCmdCopyBuffer(command_buffer,
                    buffers.tile_sort_count.deviceBuffer.buffer,
                    instance_count_readback_buffer_.buffer,
                    1,
                    &copy);
    const auto& wave_buffer = buffers.depth_wave_dispatch.deviceBuffer;
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
    vkCmdCopyBuffer(command_buffer,
                    wave_buffer.buffer,
                    instance_count_readback_buffer_.buffer,
                    1,
                    &wave_copy);
    const uint32_t armed_u32 = static_cast<uint32_t>(armed);
    validateBufferRange(instance_count_readback_buffer_,
                        2 * sizeof(uint32_t),
                        sizeof(uint32_t),
                        "depth-wave armed-count readback destination");
    vkCmdUpdateBuffer(command_buffer,
                      instance_count_readback_buffer_.buffer,
                      2 * sizeof(uint32_t),
                      sizeof(uint32_t),
                      &armed_u32);
    bufferMemoryBarrier({{instance_count_readback_buffer_, TRANSFER_WRITE}},
                        HOST_READ);
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
    vkCmdCopyBuffer(command_buffer,
                    count.buffer,
                    instance_gate_readback_buffer_.buffer,
                    1,
                    &copy);
    bufferMemoryBarrier({{instance_gate_readback_buffer_, TRANSFER_WRITE}}, HOST_READ);

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

    VkBufferCopy copy{};
    copy.srcOffset = buffers.visible_count.deviceBuffer.offset;
    copy.dstOffset = 0;
    copy.size = 2 * sizeof(uint32_t);
    validateBufferRange(count_buffer, 0, copy.size, "visible-count readback source");
    validateBufferRange(visible_count_readback_buffer_, 0, copy.size, "visible-count readback destination");
    vkCmdCopyBuffer(command_buffer,
                    buffers.visible_count.deviceBuffer.buffer,
                    visible_count_readback_buffer_.buffer,
                    1,
                    &copy);
    bufferMemoryBarrier({{visible_count_readback_buffer_, TRANSFER_WRITE}},
                        HOST_READ);
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

    bufferMemoryBarrier({{buffers.lod_gpu_counts.deviceBuffer, COMPUTE_SHADER_WRITE},
                         {buffers.lod_compact_counts.deviceBuffer, COMPUTE_SHADER_WRITE},
                         {buffers.lod_compact_protected.deviceBuffer, COMPUTE_SHADER_WRITE},
                         {buffers.lod_compact_misses.deviceBuffer, COMPUTE_SHADER_WRITE}},
                        TRANSFER_READ);

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
        vkCmdCopyBuffer(command_buffer, src.buffer,
                        lod_selection_readback_buffer_.buffer, 1, &copy);
    };
    copy_region(buffers.lod_gpu_counts.deviceBuffer, 0, 2);
    copy_region(buffers.lod_compact_counts.deviceBuffer, 2, 4);
    copy_region(buffers.lod_compact_protected.deviceBuffer, 6, kLodCompactProtectedCap);
    copy_region(buffers.lod_compact_misses.deviceBuffer, 6 + kLodCompactProtectedCap,
                2 * kLodCompactMissCap);
    bufferMemoryBarrier({{lod_selection_readback_buffer_, TRANSFER_WRITE}},
                        HOST_READ);
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
    createComputePipeline(pipeline_seed_primitive_indices, spirv_paths.at("seed_primitive_indices"));
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
    bufferMemoryBarrier({
                            {buffers.lod_logical_indices.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {chunk_to_page, TRANSFER_COMPUTE_SHADER_WRITE},
                            {out_indices, TRANSFER_COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ_WRITE);

    executeCompute(
        {{lod_count, 64}},
        &map_uniforms, sizeof(map_uniforms),
        pipeline_lod_map_indices,
        {
            buffers.lod_logical_indices.deviceBuffer,
            chunk_to_page,
            out_indices,
        });

    bufferMemoryBarrier({{out_indices, COMPUTE_SHADER_WRITE}},
                        COMPUTE_SHADER_READ);
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
    bufferMemoryBarrier({
                            {node_bounds, TRANSFER_COMPUTE_SHADER_WRITE},
                            {node_links, TRANSFER_COMPUTE_SHADER_WRITE},
                            {chunk_to_page, TRANSFER_COMPUTE_SHADER_WRITE},
                            {counts, TRANSFER_WRITE},
                            {out_indices, TRANSFER_COMPUTE_SHADER_WRITE},
                            {out_logical_indices, TRANSFER_COMPUTE_SHADER_WRITE},
                            {out_weights, TRANSFER_COMPUTE_SHADER_WRITE},
                            {chunk_touch, TRANSFER_WRITE},
                            {out_levels, TRANSFER_COMPUTE_SHADER_WRITE},
                            {page_age, TRANSFER_COMPUTE_SHADER_WRITE},
                            {page_frames, TRANSFER_COMPUTE_SHADER_WRITE},
                            {page_to_chunk, TRANSFER_COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ_WRITE);

    executeCompute(
        {{uniforms.physical_node_count, 128}},
        &uniforms, sizeof(uniforms),
        pipeline_lod_select_threshold,
        {
            node_bounds,
            node_links,
            chunk_to_page,
            counts,
            out_indices,
            out_logical_indices,
            out_weights,
            chunk_touch,
            out_levels,
            page_age,
            page_frames,
            page_to_chunk,
        });

    bufferMemoryBarrier({
                            {counts, COMPUTE_SHADER_WRITE},
                            {out_indices, COMPUTE_SHADER_WRITE},
                            {out_logical_indices, COMPUTE_SHADER_WRITE},
                            {out_weights, COMPUTE_SHADER_WRITE},
                            {out_levels, COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ);

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
    bufferMemoryBarrier({{chunk_touch, COMPUTE_SHADER_WRITE},
                         {compact_counts, TRANSFER_WRITE}},
                        COMPUTE_SHADER_READ_WRITE);
    executeCompute(
        {{uniforms.logical_chunk_count, 256}},
        &compact_uniforms, sizeof(compact_uniforms),
        pipeline_lod_compact_touch,
        {
            chunk_touch,
            compact_counts,
            compact_protected,
            compact_misses,
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

    const size_t num_splats = static_cast<size_t>(uniforms.num_splats);

    bufferMemoryBarrier({
                            {buffers.xyz_ws.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.sh0.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.shN.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.rotations.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.scaling_raw.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.opacity_raw.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {transform_indices, TRANSFER_COMPUTE_SHADER_WRITE},
                            {node_mask, TRANSFER_COMPUTE_SHADER_WRITE},
                            {overlay_params, TRANSFER_COMPUTE_SHADER_WRITE},
                            {model_transforms, TRANSFER_COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ);

    size_t alloc_size = std::max(num_splats, alloc_reserve);

    // Two-stage sort: pre-fill primitive_depth_keys with 0xFFFFFFFFu so any
    // primitive that hits an early-return path inside the projection shader
    // (z-near reject, opacity below threshold, degenerate covariance, zero
    // tiles touched) keeps the max-key sentinel and sorts to the tail.
    auto& primitive_depth_keys =
        resizeDeviceBuffer(buffers.primitive_depth_keys, alloc_size);
    bufferMemoryBarrier({{primitive_depth_keys, COMPUTE_SHADER_READ_WRITE}},
                        TRANSFER_COMPUTE_SHADER_WRITE);
    validateFillRange(primitive_depth_keys, 0, primitive_depth_keys.size, "primitive-depth sentinel fill");
    vkCmdFillBuffer(command_buffer, primitive_depth_keys.buffer,
                    primitive_depth_keys.offset, primitive_depth_keys.size,
                    0xFFFFFFFFu);
    bufferMemoryBarrier({{primitive_depth_keys, TRANSFER_COMPUTE_SHADER_WRITE}},
                        COMPUTE_SHADER_READ_WRITE);

    // Ensure transfer writes to optional LOD buffers are visible to projection.
    if (lod_indices.buffer != VK_NULL_HANDLE ||
        lod_logical_indices.buffer != VK_NULL_HANDLE ||
        lod_levels.buffer != VK_NULL_HANDLE ||
        lod_weights.buffer != VK_NULL_HANDLE) {
        std::vector<std::pair<_VulkanBuffer, BarrierMask>> barriers;
        if (lod_indices.buffer != VK_NULL_HANDLE) {
            barriers.push_back({lod_indices, TRANSFER_COMPUTE_SHADER_WRITE});
        }
        if (lod_logical_indices.buffer != VK_NULL_HANDLE) {
            barriers.push_back({lod_logical_indices, TRANSFER_COMPUTE_SHADER_WRITE});
        }
        if (lod_levels.buffer != VK_NULL_HANDLE) {
            barriers.push_back({lod_levels, TRANSFER_COMPUTE_SHADER_WRITE});
        }
        if (lod_weights.buffer != VK_NULL_HANDLE) {
            barriers.push_back({lod_weights, TRANSFER_COMPUTE_SHADER_WRITE});
        }
        bufferMemoryBarrier(barriers, COMPUTE_SHADER_READ);
    }

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

    std::vector<_VulkanBuffer> projection_buffers = {
        // inputs
        buffers.xyz_ws.deviceBuffer,
        buffers.sh0.deviceBuffer,
        buffers.shN.deviceBuffer,
        buffers.rotations.deviceBuffer,
        buffers.scaling_raw.deviceBuffer,
        buffers.opacity_raw.deviceBuffer,
        // outputs
        resizeDeviceBuffer(buffers.tiles_touched, alloc_size),
        resizeDeviceBuffer(buffers.rect_tile_space, alloc_size),
        resizeDeviceBuffer(buffers.radii, alloc_size),
        resizeDeviceBuffer(buffers.xy_vs, 2 * alloc_size),
        resizeDeviceBuffer(buffers.depths, alloc_size),
        resizeDeviceBuffer(buffers.inv_cov_vs_opacity, 4 * alloc_size),
        resizeDeviceBuffer(buffers.rgb, 3 * alloc_size),
        resizeDeviceBuffer(buffers.overlay_flags, alloc_size),
        transform_indices,
        node_mask,
        overlay_params,
        model_transforms,
        primitive_depth_keys,
        lod_indices_binding,
        lod_logical_indices_binding,
        lod_levels_binding,
        lod_weights_binding,
        lod_counts_binding,
    };

    VulkanGSRendererUniforms projection_uniforms = uniforms;
    if (buffers.quant_pool) {
        projection_uniforms.lod_page_splats = buffers.pool_page_splats;
        projection_buffers.push_back(buffers.page_frames.deviceBuffer);
    }
    executeCompute(
        {{num_splats, SUBGROUP_SIZE}},
        &projection_uniforms, sizeof(projection_uniforms),
        buffers.quant_pool
            ? (use_gut_projection ? pipeline_projection_forward_quant_3dgut
                                  : pipeline_projection_forward_quant)
            : (use_gut_projection ? pipeline_projection_forward_3dgut
                                  : pipeline_projection_forward),
        projection_buffers);
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
    const size_t num_tiles =
        static_cast<size_t>(uniforms.grid_height) * uniforms.grid_width;
    const size_t num_pixels =
        static_cast<size_t>(uniforms.image_height) * uniforms.image_width;
    if (num_tiles == 0 || num_pixels == 0)
        return;

    // This one-frame heuristic selects only the faster of two wave-correct
    // raster paths. It never sizes storage or changes the wave budget.
    const bool use_batched_raster =
        !use_gut_rasterization && !depth_capture_ &&
        buffers.num_indices >= kMinLoadBalancedRasterInstances &&
        buffers.num_indices / num_tiles >= kMinLoadBalancedAverageTileInstances;
    const size_t batch_capacity = denseTileBatchCapacity(capacity, num_tiles);

    // K/viewport allocations are established before conditional blocks.
    resizeDeviceBuffer(buffers.sorting_keys_1, capacity);
    resizeDeviceBuffer(buffers.sorting_keys_2, capacity);
    resizeDeviceBuffer(buffers.sorting_gauss_idx_1, capacity);
    resizeDeviceBuffer(buffers.sorting_gauss_idx_2, capacity);
    auto& tile_ranges = resizeDeviceBuffer(buffers.tile_ranges, num_tiles + 1u);
    auto& pixel_state = resizeDeviceBuffer(buffers.pixel_state, 4u * num_pixels);
    auto& pixel_depth = resizeDeviceBuffer(buffers.pixel_depth, num_pixels);
    auto& pixel_depth_weight =
        resizeDeviceBuffer(buffers.pixel_depth_weight, num_pixels);
    auto& n_contributors = resizeDeviceBuffer(buffers.n_contributors, num_pixels);

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
        batch_counts = resizeDeviceBuffer(buffers.tile_batch_counts, num_tiles);
        batch_offsets = resizeDeviceBuffer(buffers.tile_batch_offsets, num_tiles);
        batch_descriptors =
            resizeDeviceBuffer(buffers.tile_batch_descriptors, 4u * batch_capacity);
        batch_dispatch = resizeDeviceBuffer(
            buffers.tile_batch_dispatch_args,
            indirect::TileBatchDispatch::kLayout.word_count);
        validateIndirectLayoutBuffer(batch_dispatch,
                                     indirect::TileBatchDispatch::kLayout,
                                     "legacy tile-batch descriptor producer");
        batch_pixel_state = resizeDeviceBuffer(
            buffers.tile_batch_pixel_state,
            4u * batch_capacity * TILE_WIDTH * TILE_HEIGHT);
        batch_n_contributors = resizeDeviceBuffer(
            buffers.tile_batch_n_contributors,
            batch_capacity * TILE_WIDTH * TILE_HEIGHT);
        constexpr size_t kCumsumBlock = 1024u;
        resizeDeviceBuffer(buffers._cumsum_blockSums,
                           std::max<size_t>(1u, _CEIL_DIV(num_tiles, kCumsumBlock)));
        resizeDeviceBuffer(
            buffers._cumsum_blockSums2,
            std::max<size_t>(1u,
                             _CEIL_DIV(_CEIL_DIV(num_tiles, kCumsumBlock),
                                       kCumsumBlock)));
    }

    bufferMemoryBarrier({
        {buffers.xy_vs.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.inv_cov_vs_opacity.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.rect_tile_space.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.index_buffer_offset.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.primitive_sort_indices.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.rgb.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.depths.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.xyz_ws.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.rotations.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.scaling_raw.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.opacity_raw.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {selection_mask, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {preview_mask, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {selection_colors, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {overlay_flags, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {overlay_params, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {transform_indices, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {model_transforms, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.sorting_keys_1.deviceBuffer,
         COMPUTE_SHADER_READ_WRITE,
         COMPUTE_SHADER_READ_WRITE},
        {buffers.sorting_keys_2.deviceBuffer,
         COMPUTE_SHADER_READ_WRITE,
         COMPUTE_SHADER_READ_WRITE},
        {buffers.sorting_gauss_idx_1.deviceBuffer,
         COMPUTE_SHADER_READ_WRITE,
         COMPUTE_SHADER_READ_WRITE},
        {buffers.sorting_gauss_idx_2.deviceBuffer,
         COMPUTE_SHADER_READ_WRITE,
         COMPUTE_SHADER_READ_WRITE},
        {buffers._sorting_histogram.deviceBuffer,
         COMPUTE_SHADER_READ_WRITE,
         COMPUTE_SHADER_WRITE},
        {buffers._sorting_histogram_cumsum.deviceBuffer,
         COMPUTE_SHADER_READ_WRITE,
         COMPUTE_SHADER_WRITE},
    });

    const auto& wave_buffer = buffers.depth_wave_dispatch.deviceBuffer;
    const auto& predicate_buffer = buffers.wave_predicates.deviceBuffer;
    for (size_t wave = 0; wave < armed; ++wave) {
        const bool conditional = predicate_waves && supports_conditional_rendering_;
        const ConditionalRenderingScope conditional_scope(
            *this,
            conditional,
            vk_cmd_begin_conditional_rendering_,
            vk_cmd_end_conditional_rendering_,
            predicate_buffer,
            wave * sizeof(uint32_t));

        if (wave > 0) {
            std::vector<BufferBarrier> barriers{
                {buffers.sorting_keys_1.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_READ_WRITE},
                {buffers.sorting_keys_2.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_READ_WRITE},
                {buffers.sorting_gauss_idx_1.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_READ_WRITE},
                {buffers.sorting_gauss_idx_2.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_READ_WRITE},
                {tile_ranges, COMPUTE_SHADER_READ_WRITE, COMPUTE_SHADER_READ_WRITE},
                {pixel_state, COMPUTE_SHADER_READ_WRITE, COMPUTE_SHADER_READ_WRITE},
                {pixel_depth, COMPUTE_SHADER_READ_WRITE, COMPUTE_SHADER_READ_WRITE},
                {pixel_depth_weight,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_READ_WRITE},
                {n_contributors, COMPUTE_SHADER_READ_WRITE, COMPUTE_SHADER_READ_WRITE},
                {buffers._sorting_histogram.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_READ_WRITE},
                {buffers._sorting_histogram_cumsum.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_READ_WRITE},
            };
            if (use_batched_raster) {
                barriers.insert(barriers.end(),
                                {{batch_counts,
                                  COMPUTE_SHADER_READ_WRITE,
                                  COMPUTE_SHADER_READ_WRITE},
                                 {batch_offsets,
                                  COMPUTE_SHADER_READ_WRITE,
                                  COMPUTE_SHADER_READ_WRITE},
                                 {batch_descriptors,
                                  COMPUTE_SHADER_READ_WRITE,
                                  COMPUTE_SHADER_READ_WRITE},
                                 {batch_dispatch,
                                  COMPUTE_SHADER_READ_WRITE,
                                  COMPUTE_SHADER_READ_WRITE},
                                 {batch_pixel_state,
                                  COMPUTE_SHADER_READ_WRITE,
                                  COMPUTE_SHADER_READ_WRITE},
                                 {batch_n_contributors,
                                  COMPUTE_SHADER_READ_WRITE,
                                  COMPUTE_SHADER_READ_WRITE}});
            }
            bufferMemoryBarrier(barriers);
        }

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
        // The §5.4 idle-path lever removes the defensive sentinel prefill. The
        // partition count is exact, and keygen either emits that full interval
        // or pads its conservative culling tail before radix consumes it.
        executeComputeIndirect(record,
                               indirect::byteOffset(indirect::DepthWave::kKeygenWordOffset),
                               &wave_uniforms,
                               sizeof(wave_uniforms),
                               pipeline_generate_keys_wave,
                               {buffers.xy_vs.deviceBuffer,
                                buffers.inv_cov_vs_opacity.deviceBuffer,
                                buffers.rect_tile_space.deviceBuffer,
                                buffers.index_buffer_offset.deviceBuffer,
                                buffers.primitive_sort_indices.deviceBuffer,
                                unsorted_keys,
                                unsorted_indices,
                                wave_buffer});
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
            bufferMemoryBarrier({
                {buffers.sorted_keys().deviceBuffer,
                 COMPUTE_SHADER_WRITE,
                 COMPUTE_SHADER_READ},
                {buffers.sorted_gauss_idx().deviceBuffer,
                 COMPUTE_SHADER_WRITE,
                 COMPUTE_SHADER_READ},
            });
            executeComputeIndirect(record,
                                   indirect::byteOffset(
                                       indirect::DepthWave::kPerTileWordOffset),
                                   &wave_uniforms,
                                   sizeof(wave_uniforms),
                                   pipeline_compute_tile_ranges_and_batch_counts
                                       [buffers.is_unsorted_1],
                                   {buffers.sorted_keys().deviceBuffer,
                                    tile_ranges,
                                    count,
                                    batch_counts});
            executeCumsum(buffers,
                          buffers.tile_batch_counts,
                          buffers.tile_batch_offsets,
                          {{tile_ranges, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ}},
                          wave < HIGS_DEPTH_MAX_WAVES);
            bufferMemoryBarrier({
                {batch_offsets, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
            });
            executeComputeIndirect(record,
                                   indirect::byteOffset(
                                       indirect::DepthWave::kPerTileWordOffset),
                                   &wave_uniforms,
                                   sizeof(wave_uniforms),
                                   pipeline_tile_batch_descriptors,
                                   {tile_ranges,
                                    batch_offsets,
                                    batch_descriptors,
                                    batch_dispatch});

            auto& light_pipeline = overlays_active
                                       ? pipeline_rasterize_forward_light
                                       : pipeline_rasterize_forward_light_plain;
            executeComputeIndirect(
                record,
                indirect::byteOffset(indirect::DepthWave::kFullscreenWordOffset),
                &wave_uniforms,
                sizeof(wave_uniforms),
                light_pipeline[buffers.is_unsorted_1],
                {buffers.sorted_gauss_idx().deviceBuffer,
                 tile_ranges,
                 buffers.xy_vs.deviceBuffer,
                 buffers.inv_cov_vs_opacity.deviceBuffer,
                 buffers.rgb.deviceBuffer,
                 buffers.depths.deviceBuffer,
                 pixel_state,
                 pixel_depth,
                 n_contributors,
                 pixel_depth_weight,
                 selection_mask,
                 preview_mask,
                 selection_colors,
                 overlay_flags,
                 overlay_params});

            bufferMemoryBarrier({
                {batch_descriptors, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
                {batch_dispatch, COMPUTE_SHADER_WRITE, INDIRECT_DISPATCH_READ},
                {pixel_state, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ_WRITE},
                {pixel_depth, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ_WRITE},
                {n_contributors, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ_WRITE},
            });
            std::vector<_VulkanBuffer> batch_bindings{
                buffers.sorted_gauss_idx().deviceBuffer,
                batch_descriptors,
                buffers.xy_vs.deviceBuffer,
                buffers.inv_cov_vs_opacity.deviceBuffer,
                buffers.rgb.deviceBuffer,
                batch_pixel_state,
                batch_n_contributors,
            };
            if (overlays_active) {
                batch_bindings.insert(batch_bindings.end(),
                                      {selection_mask,
                                       preview_mask,
                                       selection_colors,
                                       overlay_flags,
                                       overlay_params});
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

            bufferMemoryBarrier({
                {batch_pixel_state, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
                {batch_n_contributors, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
            });
            std::vector<_VulkanBuffer> compose_bindings{
                buffers.sorted_gauss_idx().deviceBuffer,
                batch_descriptors,
                batch_offsets,
                buffers.xy_vs.deviceBuffer,
                buffers.inv_cov_vs_opacity.deviceBuffer,
                buffers.rgb.deviceBuffer,
                buffers.depths.deviceBuffer,
                batch_pixel_state,
                batch_n_contributors,
                pixel_state,
                pixel_depth,
                n_contributors,
            };
            if (overlays_active) {
                compose_bindings.insert(compose_bindings.end(),
                                        {selection_mask,
                                         preview_mask,
                                         selection_colors,
                                         overlay_flags,
                                         overlay_params});
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
            bufferMemoryBarrier({
                {buffers.sorted_keys().deviceBuffer,
                 COMPUTE_SHADER_WRITE,
                 COMPUTE_SHADER_READ},
                {buffers.sorted_gauss_idx().deviceBuffer,
                 COMPUTE_SHADER_WRITE,
                 COMPUTE_SHADER_READ},
            });
            executeComputeIndirect(record,
                                   indirect::byteOffset(
                                       indirect::DepthWave::kRangeWordOffset),
                                   &wave_uniforms,
                                   sizeof(wave_uniforms),
                                   pipeline_compute_tile_ranges[buffers.is_unsorted_1],
                                   {buffers.sorted_keys().deviceBuffer, tile_ranges, count});
            bufferMemoryBarrier({
                {tile_ranges, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
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
                    {buffers.sorted_gauss_idx().deviceBuffer,
                     tile_ranges,
                     buffers.xy_vs.deviceBuffer,
                     buffers.inv_cov_vs_opacity.deviceBuffer,
                     buffers.rgb.deviceBuffer,
                     buffers.depths.deviceBuffer,
                     buffers.xyz_ws.deviceBuffer,
                     buffers.rotations.deviceBuffer,
                     buffers.scaling_raw.deviceBuffer,
                     buffers.opacity_raw.deviceBuffer,
                     pixel_state,
                     pixel_depth,
                     n_contributors,
                     pixel_depth_weight,
                     selection_mask,
                     preview_mask,
                     selection_colors,
                     overlay_flags,
                     overlay_params,
                     transform_indices,
                     model_transforms});
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
                    {buffers.sorted_gauss_idx().deviceBuffer,
                     tile_ranges,
                     buffers.xy_vs.deviceBuffer,
                     buffers.inv_cov_vs_opacity.deviceBuffer,
                     buffers.rgb.deviceBuffer,
                     buffers.depths.deviceBuffer,
                     pixel_state,
                     pixel_depth,
                     n_contributors,
                     pixel_depth_weight,
                     selection_mask,
                     preview_mask,
                     selection_colors,
                     overlay_flags,
                     overlay_params});
            }
        }
    }

    if (uniforms.expected_far > 0.0f) {
        bufferMemoryBarrier({
            {pixel_depth, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ_WRITE},
            {pixel_depth_weight, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        });
        const uint32_t finalize_uniforms = static_cast<uint32_t>(num_pixels);
        executeCompute({{num_pixels, 256u}},
                       &finalize_uniforms,
                       sizeof(finalize_uniforms),
                       pipeline_expected_depth_finalize,
                       {pixel_depth, pixel_depth_weight});
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

    bufferMemoryBarrier({
                            {buffers.xyz_ws.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.rotations.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.scaling_raw.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.opacity_raw.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {transform_indices, TRANSFER_COMPUTE_SHADER_WRITE},
                            {node_mask, TRANSFER_COMPUTE_SHADER_WRITE},
                            {primitives, TRANSFER_COMPUTE_SHADER_WRITE},
                            {model_transforms, TRANSFER_COMPUTE_SHADER_WRITE},
                            {selection_out, TRANSFER_COMPUTE_SHADER_WRITE},
                            {polygon_mask, COMPUTE_SHADER_READ_WRITE},
                            {ring_pick_out, TRANSFER_COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ_WRITE);

    const size_t num_words = _CEIL_DIV(static_cast<size_t>(uniforms.num_splats), 4);
    executeCompute(
        {{num_words, SUBGROUP_SIZE}},
        &uniforms, sizeof(uniforms),
        pipeline_selection_mask,
        {
            buffers.xyz_ws.deviceBuffer,
            transform_indices,
            node_mask,
            primitives,
            model_transforms,
            buffers.rotations.deviceBuffer,
            buffers.scaling_raw.deviceBuffer,
            selection_out,
            polygon_mask,
            buffers.opacity_raw.deviceBuffer,
            ring_pick_out,
        });

    bufferMemoryBarrier({{selection_out, COMPUTE_SHADER_WRITE},
                         {ring_pick_out, COMPUTE_SHADER_WRITE}},
                        TRANSFER_READ);
}

void VulkanGSRenderer::executeSelectionPolygonRasterize(
    const VulkanGSSelectionPolygonRasterizeUniforms& uniforms,
    const _VulkanBuffer& polygon_vertices,
    const _VulkanBuffer& polygon_mask) {
    DEVICE_GUARD;

    bufferMemoryBarrier({
                            {polygon_vertices, TRANSFER_COMPUTE_SHADER_WRITE},
                            {polygon_mask, TRANSFER_COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ_WRITE);

    constexpr size_t kBlockXY = 8;
    executeCompute(
        {{static_cast<size_t>(uniforms.aabb_w), kBlockXY},
         {static_cast<size_t>(uniforms.aabb_h), kBlockXY}},
        &uniforms, sizeof(uniforms),
        pipeline_selection_polygon_rasterize,
        {
            polygon_vertices,
            polygon_mask,
        });

    bufferMemoryBarrier({{polygon_mask, COMPUTE_SHADER_WRITE}}, COMPUTE_SHADER_READ);
}

void VulkanGSRenderer::executeCumsum(
    VulkanGSPipelineBuffers& buffers,
    Buffer<int32_t>& input_buffer,
    Buffer<int32_t>& output_buffer,
    const std::vector<BufferBarrier>& additional_begin_barriers,
    const bool record_timestamps) {
    std::optional<PerfTimer::Timer<PerfTimer::_Cumsum>> timer;
    if (record_timestamps) {
        timer.emplace(this);
    }
    DEVICE_GUARD;

    size_t num_elements = input_buffer.deviceSize();
    const size_t block_0 = 1024;
    const size_t block_limit = deviceInfo.subgroupSize * deviceInfo.subgroupSize * deviceInfo.subgroupSize;
    const size_t block = std::min(block_0, block_limit);

    auto execute_cumsum_phase = [&](size_t active_elements,
                                    size_t threads_per_group,
                                    _ComputePipeline& pipeline,
                                    const std::vector<_VulkanBuffer>& phase_buffers) {
        uint32_t phase_uniforms[1] = {static_cast<uint32_t>(active_elements)};
        executeCompute(
            {{active_elements, threads_per_group}},
            phase_uniforms,
            sizeof(uint32_t),
            pipeline,
            phase_buffers);
    };

    resizeDeviceBuffer(output_buffer, num_elements);

    const auto begin_cumsum = [&](const bool uses_level_1, const bool uses_level_2) {
        std::vector<BufferBarrier> barriers = additional_begin_barriers;
        barriers.insert(barriers.end(),
                        {{input_buffer.deviceBuffer,
                          COMPUTE_SHADER_WRITE,
                          COMPUTE_SHADER_READ},
                         {output_buffer.deviceBuffer,
                          COMPUTE_SHADER_READ_WRITE,
                          COMPUTE_SHADER_WRITE}});
        if (uses_level_1) {
            barriers.push_back({buffers._cumsum_blockSums.deviceBuffer,
                                COMPUTE_SHADER_READ_WRITE,
                                COMPUTE_SHADER_WRITE});
        }
        if (uses_level_2) {
            barriers.push_back({buffers._cumsum_blockSums2.deviceBuffer,
                                COMPUTE_SHADER_READ_WRITE,
                                COMPUTE_SHADER_WRITE});
        }
        bufferMemoryBarrier(barriers);
    };

    if (num_elements <= block_0) {
        begin_cumsum(false, false);
        execute_cumsum_phase(
            num_elements, block_0,
            pipeline_cumsum.single_pass,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
            });
    }

    else if (num_elements <= block * block) {
        const size_t num_blocks = _CEIL_DIV(num_elements, block);
        resizeDeviceBuffer(buffers._cumsum_blockSums, num_blocks, true);
        begin_cumsum(true, false);

        execute_cumsum_phase(
            num_elements, block,
            pipeline_cumsum.block_scan,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        execute_cumsum_phase(
            num_blocks, block,
            pipeline_cumsum.scan_block_sums,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {output_buffer.deviceBuffer, COMPUTE_SHADER_WRITE},
                                {buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        execute_cumsum_phase(
            num_elements, block,
            pipeline_cumsum.add_block_offsets,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            });
    }

    else if (num_elements <= block * block * block) {
        const size_t num_elements_1 = _CEIL_DIV(num_elements, block);
        const size_t num_elements_2 = _CEIL_DIV(num_elements_1, block);
        resizeDeviceBuffer(buffers._cumsum_blockSums, num_elements_1, true);
        resizeDeviceBuffer(buffers._cumsum_blockSums2, num_elements_2, true);
        begin_cumsum(true, true);

        execute_cumsum_phase(
            num_elements, block,
            pipeline_cumsum.block_scan,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        execute_cumsum_phase(
            num_elements_1, block,
            pipeline_cumsum.block_scan,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                                {buffers._cumsum_blockSums2.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        execute_cumsum_phase(
            num_elements_2, block,
            pipeline_cumsum.scan_block_sums,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {buffers._cumsum_blockSums2.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        execute_cumsum_phase(
            num_elements_1, block,
            pipeline_cumsum.add_block_offsets,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {output_buffer.deviceBuffer, COMPUTE_SHADER_WRITE},
                                {buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        execute_cumsum_phase(
            num_elements, block,
            pipeline_cumsum.add_block_offsets,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
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

    bufferMemoryBarrier({
                            {buffers.index_buffer_offset.deviceBuffer, COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ);
    executeCompute(
        {{1, 1}},
        &num_splats, sizeof(num_splats),
        pipeline_prepare_tile_sort,
        {
            buffers.index_buffer_offset.deviceBuffer,
            buffers.tile_sort_count.deviceBuffer,
        });
    bufferMemoryBarrier({{buffers.tile_sort_count.deviceBuffer, COMPUTE_SHADER_WRITE}},
                        TRANSFER_COMPUTE_SHADER_READ);
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
    {
        [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".clear_histogram"));
        // The primitive-depth and tile-key radix passes reuse these workspaces
        // inside one command buffer. Finish the earlier pass before resetting
        // its global histogram and before the next upsweep overwrites the
        // partition histogram.
        if (!wave_barriers_hoisted) {
            bufferMemoryBarrier({
                {globalHistogram.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_WRITE},
                {partitionHistogram.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_WRITE},
                {buffers.unsorted_keys().deviceBuffer,
                 COMPUTE_SHADER_WRITE,
                 COMPUTE_SHADER_READ},
                {buffers.unsorted_gauss_idx().deviceBuffer,
                 COMPUTE_SHADER_WRITE,
                 COMPUTE_SHADER_READ},
            });
        }
        const uint32_t clear_uniforms[2]{
            static_cast<uint32_t>(num_passes * RADIX),
            static_cast<uint32_t>(num_parts_capacity * RADIX),
        };
        executeCompute({{1, 1}},
                       clear_uniforms,
                       sizeof(clear_uniforms),
                       pipeline_radix_histogram_clear,
                       {globalHistogram.deviceBuffer, partitionHistogram.deviceBuffer});
        std::vector<BufferBarrier> clear_barriers{
            {globalHistogram.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ_WRITE},
            {partitionHistogram.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ_WRITE},
        };
        if (wave_barriers_hoisted) {
            // The partition's record/indirect-read transition and the prior
            // wave's reuse hazards are hoisted outside this helper. Fold the
            // wave keygen -> radix dependency into the histogram-clear edge.
            clear_barriers.insert(
                clear_barriers.end(),
                {{buffers.unsorted_keys().deviceBuffer,
                  COMPUTE_SHADER_WRITE,
                  COMPUTE_SHADER_READ},
                 {buffers.unsorted_gauss_idx().deviceBuffer,
                  COMPUTE_SHADER_WRITE,
                  COMPUTE_SHADER_READ}});
        }
        bufferMemoryBarrier(clear_barriers);
    }
    if (!wave_barriers_hoisted) {
        [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".prepare_count_and_dispatch"));
        bufferMemoryBarrier({
            {count_buffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
            {dispatch_args_buffer, COMPUTE_SHADER_WRITE, INDIRECT_DISPATCH_READ},
        });
    }

    for (int pass = 0; 8 * pass < max_nonzero_bit; pass++) {
        auto& pipeline_sorting = buffers.is_unsorted_1 ? pipeline_sorting_indirect_1
                                                       : pipeline_sorting_indirect_2;

        uint32_t sort_uniforms[2];
        sort_uniforms[0] = static_cast<uint32_t>(pass);
        sort_uniforms[1] = 0;

        if (pass) {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_pingpong_barrier"));
            bufferMemoryBarrier({
                                    {buffers.unsorted_keys().deviceBuffer, COMPUTE_SHADER_WRITE},
                                    {buffers.unsorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE},
                                },
                                COMPUTE_SHADER_READ_WRITE);
        }
        {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_upsweep"));
            executeComputeIndirect(
                dispatch_args_buffer,
                indirect::byteOffset(radix_word_offset),
                sort_uniforms, 2 * sizeof(int32_t),
                pipeline_sorting.upsweep,
                {
                    buffers.unsorted_keys().deviceBuffer,
                    globalHistogram.deviceBuffer,
                    partitionHistogram.deviceBuffer,
                    count_buffer,
                });
        }

        {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_upsweep_to_spine_barrier"));
            bufferMemoryBarrier({
                                    {globalHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                                    {partitionHistogram.deviceBuffer, COMPUTE_SHADER_WRITE},
                                },
                                COMPUTE_SHADER_READ_WRITE);
        }
        {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_spine"));
            executeCompute(
                {{RADIX, 1}},
                sort_uniforms, 2 * sizeof(int32_t),
                pipeline_sorting.spine,
                {
                    globalHistogram.deviceBuffer,
                    partitionHistogram.deviceBuffer,
                    count_buffer,
                });
        }

        {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_spine_to_downsweep_barrier"));
            bufferMemoryBarrier({
                                    {globalHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                                    {partitionHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                                },
                                COMPUTE_SHADER_READ);
        }
        {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_downsweep"));
            executeComputeIndirect(
                dispatch_args_buffer,
                indirect::byteOffset(radix_word_offset),
                sort_uniforms, 2 * sizeof(int32_t),
                pipeline_sorting.downsweep,
                {
                    globalHistogram.deviceBuffer,
                    partitionHistogram.deviceBuffer,
                    buffers.unsorted_keys().deviceBuffer,
                    buffers.unsorted_gauss_idx().deviceBuffer,
                    buffers.sorted_keys().deviceBuffer,
                    buffers.sorted_gauss_idx().deviceBuffer,
                    count_buffer,
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

    {
        PerfTimer::Timer<PerfTimer::BuildVisibleFlags> gpu_timer(this);
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.build_visible_flags");
        bufferMemoryBarrier({
                                {buffers.tiles_touched.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ);
        executeCompute(
            {{num_splats, 64}},
            &visible_uniforms, sizeof(visible_uniforms),
            pipeline_visible_flags,
            {
                buffers.tiles_touched.deviceBuffer,
                buffers.visible_flags.deviceBuffer,
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
        bufferMemoryBarrier({
                                {buffers.visible_prefix.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ);
        executeCompute(
            {{1, 1}},
            &prepare_uniforms, sizeof(prepare_uniforms),
            pipeline_prepare_visible_sort,
            {
                buffers.visible_prefix.deviceBuffer,
                buffers.visible_count.deviceBuffer,
                buffers.visible_sort_dispatch_args.deviceBuffer,
            });
        bufferMemoryBarrier({
                                {buffers.visible_count.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            TRANSFER_COMPUTE_SHADER_READ);
        recordVisibleCountReadback(buffers, num_splats);
    }

    {
        PerfTimer::Timer<PerfTimer::CompactVisiblePrimitives> gpu_timer(this);
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.compact_visible_primitives");
        bufferMemoryBarrier({
                                {buffers.primitive_depth_keys.deviceBuffer, COMPUTE_SHADER_WRITE},
                                {buffers.visible_prefix.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ);
        executeCompute(
            {{num_splats, 64}},
            &visible_uniforms, sizeof(visible_uniforms),
            pipeline_compact_visible_primitives,
            {
                buffers.tiles_touched.deviceBuffer,
                buffers.visible_prefix.deviceBuffer,
                buffers.primitive_depth_keys.deviceBuffer,
                *unsorted_keys,
                *unsorted_idx,
            });
    }

    {
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.sort_visible_primitives");
        bufferMemoryBarrier({
                                {*unsorted_keys, COMPUTE_SHADER_WRITE},
                                {*unsorted_idx, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);

        // Stage 1 sort: num_bits=32 is intentional. Projection writes the full
        // float-as-uint bit pattern of a non-negative radial-distance key, whose
        // unsigned ordering is monotonic. This visible layout contains only the
        // radix command; range construction belongs to the later tile layout.
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
        bufferMemoryBarrier({{buffers.sorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE}},
                            TRANSFER_READ);
        bufferMemoryBarrier({{sort_indices, COMPUTE_SHADER_READ}},
                            TRANSFER_WRITE);
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
        vkCmdCopyBuffer(command_buffer,
                        buffers.sorted_gauss_idx().deviceBuffer.buffer,
                        sort_indices.buffer, 1, &copy);
        bufferMemoryBarrier({{sort_indices, TRANSFER_WRITE}},
                            COMPUTE_SHADER_READ);
    }
}

void VulkanGSRenderer::executeApplyDepthOrdering(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers) {
    PerfTimer::Timer<PerfTimer::ApplyDepthOrdering> timer(this);
    DEVICE_GUARD;

    const size_t num_splats = static_cast<size_t>(uniforms.num_splats);
    if (num_splats == 0)
        return;

    auto& tiles_touched_ordered =
        resizeDeviceBuffer(buffers.tiles_touched_depth_ordered, num_splats);

    bufferMemoryBarrier({{buffers.primitive_sort_indices.deviceBuffer, TRANSFER_WRITE},
                         {buffers.tiles_touched.deviceBuffer, COMPUTE_SHADER_WRITE},
                         {buffers.visible_count.deviceBuffer, COMPUTE_SHADER_WRITE}},
                        COMPUTE_SHADER_READ);

    struct ApplyUniforms {
        uint32_t num_splats;
        uint32_t pad0, pad1, pad2;
    } apply_uniforms{static_cast<uint32_t>(num_splats), 0, 0, 0};

    executeCompute(
        {{num_splats, 64}},
        &apply_uniforms, sizeof(apply_uniforms),
        pipeline_apply_depth_ordering,
        {
            buffers.primitive_sort_indices.deviceBuffer,
            buffers.tiles_touched.deviceBuffer,
            tiles_touched_ordered,
            buffers.visible_count.deviceBuffer,
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

    const size_t num_splats = static_cast<size_t>(uniforms.num_splats);
    if (num_splats == 0)
        return;

    bufferMemoryBarrier({
                            {buffers.xyz_ws.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {transform_indices, TRANSFER_COMPUTE_SHADER_WRITE},
                            {node_mask, TRANSFER_COMPUTE_SHADER_WRITE},
                            {overlay_params, TRANSFER_COMPUTE_SHADER_WRITE},
                            {model_transforms, TRANSFER_COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ);

    auto& survivors = resizeDeviceBuffer(buffers.survivors, num_splats);
    auto& survivor_state = clearDeviceBuffer(
        buffers.survivor_state,
        indirect::SurvivorState::kLayout.word_count);
    validateIndirectLayoutBuffer(survivor_state,
                                 indirect::SurvivorState::kLayout,
                                 "cull_prepare survivor-state producer");
    auto& emit_count = resizeDeviceBuffer(buffers.visible_emit_count, 1);
    bufferMemoryBarrier({{survivor_state, TRANSFER_WRITE}}, COMPUTE_SHADER_READ_WRITE);

    if (lod_indices.buffer != VK_NULL_HANDLE ||
        lod_logical_indices.buffer != VK_NULL_HANDLE) {
        std::vector<std::pair<_VulkanBuffer, BarrierMask>> barriers;
        if (lod_indices.buffer != VK_NULL_HANDLE) {
            barriers.push_back({lod_indices, TRANSFER_COMPUTE_SHADER_WRITE});
        }
        if (lod_logical_indices.buffer != VK_NULL_HANDLE) {
            barriers.push_back({lod_logical_indices, TRANSFER_COMPUTE_SHADER_WRITE});
        }
        bufferMemoryBarrier(barriers, COMPUTE_SHADER_READ);
    }

    const _VulkanBuffer lod_indices_binding =
        (lod_indices.buffer != VK_NULL_HANDLE) ? lod_indices : survivor_state;
    const _VulkanBuffer lod_logical_indices_binding =
        (lod_logical_indices.buffer != VK_NULL_HANDLE) ? lod_logical_indices : lod_indices_binding;
    const _VulkanBuffer lod_counts_binding =
        (lod_counts.buffer != VK_NULL_HANDLE) ? lod_counts : survivor_state;

    executeCompute(
        {{num_splats, 256}},
        &uniforms, sizeof(uniforms),
        pipeline_cull_splats,
        {
            buffers.xyz_ws.deviceBuffer,
            transform_indices,
            node_mask,
            overlay_params,
            model_transforms,
            lod_indices_binding,
            lod_logical_indices_binding,
            lod_counts_binding,
            survivors,
            survivor_state,
        });

    bufferMemoryBarrier({{survivor_state, COMPUTE_SHADER_WRITE}}, COMPUTE_SHADER_READ_WRITE);
    executeCompute(
        {{1, 1}},
        nullptr, 0,
        pipeline_cull_prepare,
        {
            survivor_state,
            emit_count,
        });

    bufferMemoryBarrier({{survivor_state, COMPUTE_SHADER_WRITE}}, INDIRECT_DISPATCH_READ);
    bufferMemoryBarrier({
                            {survivors, COMPUTE_SHADER_WRITE},
                            {emit_count, COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ_WRITE);
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

    if (visible_capacity == 0)
        return;

    bufferMemoryBarrier({
                            {buffers.sh0.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.shN.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.rotations.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.scaling_raw.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.opacity_raw.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ);

    if (lod_levels.buffer != VK_NULL_HANDLE || lod_weights.buffer != VK_NULL_HANDLE) {
        std::vector<std::pair<_VulkanBuffer, BarrierMask>> barriers;
        if (lod_levels.buffer != VK_NULL_HANDLE) {
            barriers.push_back({lod_levels, TRANSFER_COMPUTE_SHADER_WRITE});
        }
        if (lod_weights.buffer != VK_NULL_HANDLE) {
            barriers.push_back({lod_weights, TRANSFER_COMPUTE_SHADER_WRITE});
        }
        bufferMemoryBarrier(barriers, COMPUTE_SHADER_READ);
    }

    VulkanGSRendererUniforms survivor_uniforms = uniforms;
    survivor_uniforms.sort_capacity = static_cast<uint32_t>(
        std::min<size_t>(visible_capacity,
                         static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
    if (buffers.quant_pool) {
        survivor_uniforms.lod_page_splats = buffers.pool_page_splats;
    }

    auto& unsorted_keys = resizeDeviceBuffer(buffers.unsorted_keys(), visible_capacity);
    auto& unsorted_idx = resizeDeviceBuffer(buffers.unsorted_gauss_idx(), visible_capacity);

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

    std::vector<_VulkanBuffer> projection_buffers = {
        // inputs
        buffers.xyz_ws.deviceBuffer,
        buffers.sh0.deviceBuffer,
        buffers.shN.deviceBuffer,
        buffers.rotations.deviceBuffer,
        buffers.scaling_raw.deviceBuffer,
        buffers.opacity_raw.deviceBuffer,
        // compact-slot outputs (slots 6 and 8 are absent from the
        // pipeline layout; the entries are placeholders)
        unsorted_keys,
        resizeDeviceBuffer(buffers.rect_tile_space, visible_capacity),
        unsorted_keys,
        resizeDeviceBuffer(buffers.xy_vs, 2 * visible_capacity),
        resizeDeviceBuffer(buffers.depths, visible_capacity),
        resizeDeviceBuffer(buffers.inv_cov_vs_opacity, 4 * visible_capacity),
        resizeDeviceBuffer(buffers.rgb, 3 * visible_capacity),
        resizeDeviceBuffer(buffers.overlay_flags, visible_capacity),
        transform_indices,
        node_mask,
        overlay_params,
        model_transforms,
        unsorted_keys,
        lod_indices_binding,
        lod_logical_indices_binding,
        lod_levels_binding,
        lod_weights_binding,
        lod_counts_binding,
        buffers.survivors.deviceBuffer,
        buffers.survivor_state.deviceBuffer,
        unsorted_idx,
        buffers.visible_emit_count.deviceBuffer,
        resizeDeviceBuffer(buffers.orig_ids, visible_capacity),
    };
    if (buffers.quant_pool) {
        projection_buffers.push_back(buffers.page_frames.deviceBuffer);
    }
    executeComputeIndirect(
        buffers.survivor_state.deviceBuffer,
        indirect::byteOffset(indirect::SurvivorState::kProjectionWordOffset),
        &survivor_uniforms, sizeof(survivor_uniforms),
        buffers.quant_pool ? pipeline_projection_forward_quant_survivors
                           : pipeline_projection_forward_survivors,
        projection_buffers);
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

    {
        PerfTimer::Timer<PerfTimer::PrepareVisibleSort> gpu_timer(this);
        bufferMemoryBarrier({
                                {buffers.visible_emit_count.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ);
        executeCompute(
            {{1, 1}},
            &prepare_uniforms, sizeof(prepare_uniforms),
            pipeline_prepare_visible_chain,
            {
                buffers.visible_emit_count.deviceBuffer,
                buffers.visible_count.deviceBuffer,
                visible_dispatch,
                cumsum_counts,
            });
        bufferMemoryBarrier({
                                {buffers.visible_count.deviceBuffer, COMPUTE_SHADER_WRITE},
                                {cumsum_counts, COMPUTE_SHADER_WRITE},
                            },
                            TRANSFER_COMPUTE_SHADER_READ);
        bufferMemoryBarrier({{visible_dispatch, COMPUTE_SHADER_WRITE}},
                            INDIRECT_DISPATCH_READ);
        // The readback's bound must be the render domain: clamping the raw
        // count to the frame's *capacity* would mask exactly the clamping the
        // raw count exists to detect.
        recordVisibleCountReadback(buffers, static_cast<size_t>(uniforms.num_splats));
    }

    {
        bufferMemoryBarrier({
                                {buffers.unsorted_keys().deviceBuffer, COMPUTE_SHADER_WRITE},
                                {buffers.unsorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
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
        bufferMemoryBarrier({{buffers.sorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE}},
                            COMPUTE_SHADER_READ);
        executeComputeIndirect(
            visible_dispatch,
            indirect::byteOffset(indirect::VisibleChainDispatch::kPerElementWordOffset),
            &copy_uniforms, sizeof(copy_uniforms),
            pipeline_copy_visible_indices,
            {
                buffers.sorted_gauss_idx().deviceBuffer,
                sort_indices,
                buffers.visible_count.deviceBuffer,
            });
        bufferMemoryBarrier({{sort_indices, COMPUTE_SHADER_WRITE}},
                            COMPUTE_SHADER_READ);
    }
}

void VulkanGSRenderer::executeMacroCoverage(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    size_t visible_capacity) {
    PerfTimer::Timer<PerfTimer::ApplyDepthOrdering> timer(this);
    DEVICE_GUARD;

    if (visible_capacity == 0)
        return;

    // Reuses tiles_touched_depth_ordered as the per-rank macro coverage
    // counts; the visible-bounded cumsum into index_buffer_offset is shared
    // with the render-tile chain.
    auto& macro_counts =
        resizeDeviceBuffer(buffers.tiles_touched_depth_ordered, visible_capacity);

    bufferMemoryBarrier({{buffers.rect_tile_space.deviceBuffer, COMPUTE_SHADER_WRITE}},
                        COMPUTE_SHADER_READ);

    executeComputeIndirect(
        buffers.visible_dispatch.deviceBuffer,
        indirect::byteOffset(indirect::VisibleChainDispatch::kPerElementWordOffset),
        &uniforms, sizeof(uniforms),
        pipeline_macro_coverage,
        {
            buffers.primitive_sort_indices.deviceBuffer,
            buffers.rect_tile_space.deviceBuffer,
            macro_counts,
            buffers.visible_count.deviceBuffer,
            buffers.xy_vs.deviceBuffer,
            buffers.inv_cov_vs_opacity.deviceBuffer,
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
    const size_t num_macro =
        _CEIL_DIV(static_cast<size_t>(uniforms.grid_width), size_t{HIGS_MACRO_T16_W}) *
        _CEIL_DIV(static_cast<size_t>(uniforms.grid_height), size_t{HIGS_MACRO_T16_H});
    const size_t num_pixels =
        static_cast<size_t>(uniforms.image_height) * uniforms.image_width;
    if (num_macro == 0 || num_pixels == 0)
        return;

    const size_t max_batches =
        _CEIL_DIV(capacity, size_t{RASTER_BATCH_SIZE}) + num_macro;
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

    // Every allocation below is content-independent: K, viewport geometry, or
    // a fixed indirect-layout size. Do this before opening conditional blocks.
    resizeDeviceBuffer(buffers.sorting_keys_1, capacity);
    resizeDeviceBuffer(buffers.sorting_keys_2, capacity);
    resizeDeviceBuffer(buffers.sorting_gauss_idx_1, capacity);
    resizeDeviceBuffer(buffers.sorting_gauss_idx_2, capacity);
    auto& tile_ranges = resizeDeviceBuffer(buffers.tile_ranges, num_macro + 1u);
    auto& batch_counts = resizeDeviceBuffer(buffers.tile_batch_counts, num_macro);
    auto& batch_offsets = resizeDeviceBuffer(buffers.tile_batch_offsets, num_macro);
    auto& macro_wave_args = resizeDeviceBuffer(
        buffers.macro_wave_args,
        indirect::MacroWaveDispatch::kLayout.word_count);
    validateIndirectLayoutBuffer(macro_wave_args,
                                 indirect::MacroWaveDispatch::kLayout,
                                 "macro_batch_prepare producer");
    auto& partials = resizeDeviceBuffer(
        buffers.macro_partials,
        pool_batches * HIGS_MACRO_TILE_SIZE_TILES * HIGS_TILE_SIZE * 4u);
    auto& active_mask = resizeDeviceBuffer(buffers.macro_active_mask, max_batches);
    auto& pixel_state = resizeDeviceBuffer(buffers.pixel_state, 4u * num_pixels);
    auto& pixel_depth = resizeDeviceBuffer(buffers.pixel_depth, num_pixels);
    auto& n_contributors = resizeDeviceBuffer(buffers.n_contributors, num_pixels);

    constexpr size_t kRadix = 256u;
    constexpr size_t kPartitionSize = 512u * 8u;
    const size_t radix_passes = _CEIL_DIV(static_cast<size_t>(sort_bits), size_t{8});
    resizeDeviceBuffer(buffers._sorting_histogram, radix_passes * kRadix);
    resizeDeviceBuffer(buffers._sorting_histogram_cumsum,
                       _CEIL_DIV(capacity, kPartitionSize) * kRadix);
    constexpr size_t kCumsumBlock = 1024u;
    resizeDeviceBuffer(buffers._cumsum_blockSums,
                       std::max<size_t>(1u, _CEIL_DIV(num_macro, kCumsumBlock)));
    resizeDeviceBuffer(
        buffers._cumsum_blockSums2,
        std::max<size_t>(1u,
                         _CEIL_DIV(_CEIL_DIV(num_macro, kCumsumBlock), kCumsumBlock)));

    const bool use_fp32 = overlays_active || (uniforms.lod_enabled & 4u) != 0u;
    auto& raster_pipeline = overlays_active
                                ? pipeline_macro_raster_overlays
                                : (use_fp32 ? pipeline_macro_raster_fp32
                                            : pipeline_macro_raster);
    auto& compose_pipeline = overlays_active
                                 ? pipeline_macro_compose_overlays
                                 : pipeline_macro_compose;

    bufferMemoryBarrier({
        {buffers.xy_vs.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.inv_cov_vs_opacity.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.rect_tile_space.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.index_buffer_offset.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.primitive_sort_indices.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.visible_count.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.rgb.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.depths.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {selection_mask, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {preview_mask, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {selection_colors, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {overlay_params, TRANSFER_COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.sorting_keys_1.deviceBuffer,
         COMPUTE_SHADER_READ_WRITE,
         COMPUTE_SHADER_READ_WRITE},
        {buffers.sorting_keys_2.deviceBuffer,
         COMPUTE_SHADER_READ_WRITE,
         COMPUTE_SHADER_READ_WRITE},
        {buffers.sorting_gauss_idx_1.deviceBuffer,
         COMPUTE_SHADER_READ_WRITE,
         COMPUTE_SHADER_READ_WRITE},
        {buffers.sorting_gauss_idx_2.deviceBuffer,
         COMPUTE_SHADER_READ_WRITE,
         COMPUTE_SHADER_READ_WRITE},
        {buffers._sorting_histogram.deviceBuffer,
         COMPUTE_SHADER_READ_WRITE,
         COMPUTE_SHADER_WRITE},
        {buffers._sorting_histogram_cumsum.deviceBuffer,
         COMPUTE_SHADER_READ_WRITE,
         COMPUTE_SHADER_WRITE},
    });

    const auto& wave_buffer = buffers.depth_wave_dispatch.deviceBuffer;
    const auto& predicate_buffer = buffers.wave_predicates.deviceBuffer;
    for (size_t wave = 0; wave < armed; ++wave) {
        const bool conditional = predicate_waves && supports_conditional_rendering_;
        const ConditionalRenderingScope conditional_scope(
            *this,
            conditional,
            vk_cmd_begin_conditional_rendering_,
            vk_cmd_end_conditional_rendering_,
            predicate_buffer,
            wave * sizeof(uint32_t));

        if (wave > 0) {
            bufferMemoryBarrier({
                {buffers.sorting_keys_1.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_READ_WRITE},
                {buffers.sorting_keys_2.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_READ_WRITE},
                {buffers.sorting_gauss_idx_1.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_READ_WRITE},
                {buffers.sorting_gauss_idx_2.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_READ_WRITE},
                {tile_ranges, COMPUTE_SHADER_READ_WRITE, COMPUTE_SHADER_READ_WRITE},
                {batch_counts, COMPUTE_SHADER_READ_WRITE, COMPUTE_SHADER_READ_WRITE},
                {batch_offsets, COMPUTE_SHADER_READ_WRITE, COMPUTE_SHADER_READ_WRITE},
                {macro_wave_args, COMPUTE_SHADER_READ_WRITE, COMPUTE_SHADER_READ_WRITE},
                {partials, COMPUTE_SHADER_READ_WRITE, COMPUTE_SHADER_READ_WRITE},
                {active_mask, COMPUTE_SHADER_READ_WRITE, COMPUTE_SHADER_READ_WRITE},
                {pixel_state, COMPUTE_SHADER_READ_WRITE, COMPUTE_SHADER_READ_WRITE},
                {pixel_depth, COMPUTE_SHADER_READ_WRITE, COMPUTE_SHADER_READ_WRITE},
                {buffers._sorting_histogram.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_READ_WRITE},
                {buffers._sorting_histogram_cumsum.deviceBuffer,
                 COMPUTE_SHADER_READ_WRITE,
                 COMPUTE_SHADER_READ_WRITE},
            });
        }

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
        // See the legacy loop: exact emission/padding replaces the defensive
        // sentinel prefill and saves one per-wave dependency barrier.
        executeComputeIndirect(
            record,
            indirect::byteOffset(indirect::DepthWave::kKeygenWordOffset),
            &wave_uniforms,
            sizeof(wave_uniforms),
            pipeline_generate_macro_keys_wave,
            {buffers.xy_vs.deviceBuffer,
             buffers.inv_cov_vs_opacity.deviceBuffer,
             buffers.rect_tile_space.deviceBuffer,
             buffers.index_buffer_offset.deviceBuffer,
             buffers.primitive_sort_indices.deviceBuffer,
             unsorted_keys,
             unsorted_indices,
             buffers.visible_count.deviceBuffer,
             wave_buffer});

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

        bufferMemoryBarrier({
            {buffers.sorted_keys().deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
            {buffers.sorted_gauss_idx().deviceBuffer,
             COMPUTE_SHADER_WRITE,
             COMPUTE_SHADER_READ},
        });
        executeComputeIndirect(
            record,
            indirect::byteOffset(indirect::DepthWave::kPerTileWordOffset),
            &wave_uniforms,
            sizeof(wave_uniforms),
            pipeline_compute_macro_ranges[buffers.is_unsorted_1],
            {buffers.sorted_keys().deviceBuffer, tile_ranges, count, batch_counts});

        executeCumsum(buffers,
                      buffers.tile_batch_counts,
                      buffers.tile_batch_offsets,
                      {{tile_ranges, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ}},
                      wave < HIGS_DEPTH_MAX_WAVES);
        bufferMemoryBarrier({
            {batch_offsets, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        });
        executeCompute({{1, 1}},
                       &wave_uniforms,
                       sizeof(wave_uniforms),
                       pipeline_macro_batch_prepare,
                       {batch_offsets, macro_wave_args});
        bufferMemoryBarrier({
            {macro_wave_args, COMPUTE_SHADER_WRITE, INDIRECT_DISPATCH_READ},
        });

        std::vector<_VulkanBuffer> raster_bindings{
            buffers.sorted_gauss_idx().deviceBuffer,
            tile_ranges,
            batch_offsets,
            buffers.xy_vs.deviceBuffer,
            buffers.inv_cov_vs_opacity.deviceBuffer,
            buffers.rgb.deviceBuffer,
            partials,
            active_mask,
        };
        if (overlays_active) {
            raster_bindings.insert(raster_bindings.end(),
                                   {selection_mask,
                                    preview_mask,
                                    selection_colors,
                                    buffers.overlay_flags.deviceBuffer,
                                    overlay_params,
                                    buffers.orig_ids.deviceBuffer});
        }
        std::vector<_VulkanBuffer> compose_bindings{
            buffers.sorted_gauss_idx().deviceBuffer,
            tile_ranges,
            batch_offsets,
            buffers.xy_vs.deviceBuffer,
            buffers.inv_cov_vs_opacity.deviceBuffer,
            buffers.rgb.deviceBuffer,
            buffers.depths.deviceBuffer,
            partials,
            active_mask,
            pixel_state,
            pixel_depth,
            n_contributors,
        };
        if (overlays_active) {
            compose_bindings.insert(compose_bindings.end(),
                                    {selection_mask,
                                     preview_mask,
                                     selection_colors,
                                     buffers.overlay_flags.deviceBuffer,
                                     overlay_params,
                                     buffers.orig_ids.deviceBuffer});
        }

        for (size_t batch_wave = 0; batch_wave < batch_waves; ++batch_wave) {
            wave_uniforms.wave_base =
                static_cast<uint32_t>(batch_wave * HIGS_RASTER_WAVE_BATCHES);
            if (batch_wave > 0) {
                bufferMemoryBarrier({
                    {partials, COMPUTE_SHADER_READ, COMPUTE_SHADER_READ_WRITE},
                    {pixel_state, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ_WRITE},
                    {pixel_depth, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ_WRITE},
                });
            }
            executeComputeIndirect(
                macro_wave_args,
                indirect::byteOffset(indirect::MacroWaveDispatch::rasterWordOffset(batch_wave)),
                &wave_uniforms,
                sizeof(wave_uniforms),
                raster_pipeline[buffers.is_unsorted_1],
                raster_bindings);
            bufferMemoryBarrier({
                {partials, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
                {active_mask, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
            });
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

    auto& input = buffers.tiles_touched_depth_ordered.deviceBuffer;
    auto& output = resizeDeviceBuffer(buffers.index_buffer_offset, visible_capacity);
    auto& block_sums = resizeDeviceBuffer(buffers._cumsum_blockSums, c1_capacity, true);
    auto& block_sums2 = resizeDeviceBuffer(buffers._cumsum_blockSums2, c2_capacity, true);
    auto& counts = buffers.cumsum_counts.deviceBuffer;
    auto& dispatch = buffers.visible_dispatch.deviceBuffer;

    const auto level_uniform = [](uint32_t level) { return level; };

    bufferMemoryBarrier({{input, COMPUTE_SHADER_WRITE}}, COMPUTE_SHADER_READ);

    {
        PerfTimer::Timer<PerfTimer::_Cumsum> cumsum_timer(this);

        // Always-recorded 3-level indirect scan. Degenerate levels dispatch a
        // single group over 1 element, so no host-side branching on the count.
        uint32_t level = level_uniform(0);
        executeComputeIndirect(dispatch,
                               indirect::byteOffset(indirect::VisibleChainDispatch::kCumsumLevel0WordOffset),
                               &level, sizeof(level),
                               pipeline_cumsum_indirect.block_scan,
                               {input, output, block_sums, counts});

        bufferMemoryBarrier({{block_sums, COMPUTE_SHADER_WRITE}}, COMPUTE_SHADER_READ_WRITE);
        level = level_uniform(1);
        executeComputeIndirect(dispatch,
                               indirect::byteOffset(indirect::VisibleChainDispatch::kCumsumLevel1WordOffset),
                               &level, sizeof(level),
                               pipeline_cumsum_indirect.block_scan,
                               {block_sums, block_sums, block_sums2, counts});

        bufferMemoryBarrier({
                                {block_sums, COMPUTE_SHADER_READ_WRITE},
                                {block_sums2, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        level = level_uniform(2);
        executeCompute({{1, 1}},
                       &level, sizeof(level),
                       pipeline_cumsum_indirect.scan_block_sums,
                       {block_sums, block_sums, block_sums2, counts});

        bufferMemoryBarrier({{block_sums2, COMPUTE_SHADER_READ_WRITE}},
                            COMPUTE_SHADER_READ_WRITE);
        level = level_uniform(1);
        executeComputeIndirect(dispatch,
                               indirect::byteOffset(indirect::VisibleChainDispatch::kCumsumLevel1WordOffset),
                               &level, sizeof(level),
                               pipeline_cumsum_indirect.add_block_offsets,
                               {block_sums, block_sums, block_sums2, counts});

        bufferMemoryBarrier({
                                {output, COMPUTE_SHADER_WRITE},
                                {block_sums, COMPUTE_SHADER_READ_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        level = level_uniform(0);
        executeComputeIndirect(dispatch,
                               indirect::byteOffset(indirect::VisibleChainDispatch::kCumsumLevel0WordOffset),
                               &level, sizeof(level),
                               pipeline_cumsum_indirect.add_block_offsets,
                               {input, output, block_sums, counts});
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

        bufferMemoryBarrier({{output, COMPUTE_SHADER_WRITE}}, COMPUTE_SHADER_READ);
        executeCompute(
            {{1, 1}},
            &visible_limit, sizeof(visible_limit),
            pipeline_prepare_tile_sort_visible,
            {
                output,
                buffers.tile_sort_count.deviceBuffer,
                buffers.visible_count.deviceBuffer,
            });
    }

    bufferMemoryBarrier({{buffers.tile_sort_count.deviceBuffer, COMPUTE_SHADER_WRITE}},
                        TRANSFER_COMPUTE_SHADER_READ);
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

    VulkanGSRendererUniforms partition_uniforms = uniforms;
    partition_uniforms.sort_capacity = HIGS_DEPTH_WAVE_INSTANCES;
    partition_uniforms.depth_wave = static_cast<uint32_t>(armed);
    bufferMemoryBarrier({
        {buffers.index_buffer_offset.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
        {buffers.tile_sort_count.deviceBuffer, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ},
    });
    std::vector<_VulkanBuffer> bindings{buffers.index_buffer_offset.deviceBuffer};
    if (visible_bounded)
        bindings.push_back(buffers.visible_count.deviceBuffer);
    bindings.insert(bindings.end(),
                    {buffers.tile_sort_count.deviceBuffer, wave_dispatch, predicates});
    executeCompute({{1, 1}},
                   &partition_uniforms,
                   sizeof(partition_uniforms),
                   visible_bounded ? pipeline_wave_partition_visible
                                   : pipeline_wave_partition,
                   bindings);

    std::vector<BufferBarrier> post_barriers{
        {wave_dispatch, COMPUTE_SHADER_WRITE, TRANSFER_COMPUTE_SHADER_INDIRECT_READ},
        {buffers.tile_sort_count.deviceBuffer,
         COMPUTE_SHADER_WRITE,
         TRANSFER_COMPUTE_SHADER_READ},
    };
    if (supports_conditional_rendering_) {
        post_barriers.push_back(
            {predicates, COMPUTE_SHADER_WRITE, CONDITIONAL_RENDERING_READ});
    }
    bufferMemoryBarrier(post_barriers);
    recordInstanceCountReadback(buffers, armed);
}
