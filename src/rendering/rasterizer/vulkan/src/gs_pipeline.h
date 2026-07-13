#pragma once

#include <algorithm> // std::sort
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring> // memcpy
#include <exception>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cassert>

#include "buffer.h"
#include "rendering/vulkan_result.hpp"

class VulkanGSPipeline {
public:
    using TimerCallback = std::function<void(const std::vector<std::pair<size_t, double>>&)>;
    using CpuTimerCallback = std::function<void(std::string_view, double)>;

    VulkanGSPipeline();
    ~VulkanGSPipeline() noexcept;

    void initializeExternal(VkInstance external_instance,
                            VkPhysicalDevice external_physical_device,
                            VkDevice external_device,
                            VkQueue external_queue,
                            uint32_t external_queue_family_index,
                            VmaAllocator external_allocator,
                            VkPipelineCache external_pipeline_cache = VK_NULL_HANDLE);
    void cleanup();
    void cleanupBuffers(VulkanGSPipelineBuffers& buffers);
    void assignBufferLabels(VulkanGSPipelineBuffers& buffers);

    void createBuffer(size_t size, _VulkanBuffer& buffer);
    void destroyBuffer(_VulkanBuffer& buffer);
    void resizeDeviceBuffer(_VulkanBuffer& deviceBuffer, size_t new_byte_size, bool no_shrink = true);
    template <typename T>
    _VulkanBuffer& resizeDeviceBuffer(Buffer<T>& buffer, size_t new_size, bool no_shrink = true);
    template <typename T>
    _VulkanBuffer& clearDeviceBuffer(Buffer<T>& buffer, size_t new_size);
    template <typename T>
    _VulkanBuffer& resizeAndCopyDeviceBuffer(Buffer<T>& buffer, size_t new_size, bool clear);
    void beginCommandBatch();
    void endCommandBatch(bool use_fence = true,
                         VkSemaphore signal_semaphore = VK_NULL_HANDLE,
                         std::uint64_t signal_value = 0,
                         VkSemaphore secondary_signal_semaphore = VK_NULL_HANDLE,
                         std::uint64_t secondary_signal_value = 0);
    void cancelCommandBatch() noexcept;
    void waitForPendingBatch();
    [[nodiscard]] bool timelineValueComplete(VkSemaphore semaphore, std::uint64_t value) const;
    // Pure host-side submission evidence. Updated immediately after
    // vkQueueSubmit succeeds, before any post-submit bookkeeping can throw.
    [[nodiscard]] bool wasTimelineSignalSubmitted(VkSemaphore semaphore,
                                                  std::uint64_t value) const noexcept;
    void addTimelineWait(VkSemaphore semaphore, std::uint64_t value, VkPipelineStageFlags stage_mask);
    bool isCommandBatchInProgress() const {
        return commandBatchInProgress;
    }
    VkCommandBuffer activeCommandBuffer() const {
        return command_buffer;
    }
    bool writeTimestamp(int delta);
    bool writeTimestampNoExcept(int delta);
    void addTimerCallback(TimerCallback callback);
    void setCpuTimerCallback(CpuTimerCallback callback);

    size_t getCurrentAllocSize() const { return current_vram; }
    size_t getPeakAllocSize() const { return peak_vram; }

    enum BarrierMask {
        TRANSFER_READ,
        TRANSFER_WRITE,
        TRANSFER_READ_WRITE,
        COMPUTE_SHADER_READ,
        COMPUTE_SHADER_WRITE,
        COMPUTE_SHADER_READ_WRITE,
        TRANSFER_COMPUTE_SHADER_READ,
        TRANSFER_COMPUTE_SHADER_WRITE,
        TRANSFER_COMPUTE_SHADER_READ_WRITE,
        HOST_READ,
        HOST_WRITE,
        HOST_READ_WRITE,
        INDIRECT_DISPATCH_READ,
    };

protected:
    bool commandBatchInProgress = false;
    uint32_t timestampNumWritten = 0;
    uint32_t timestampStackDepth = 0;
    std::vector<TimerCallback> timerCallbacks;
    CpuTimerCallback cpuTimerCallback_;

    class [[nodiscard]] CpuStageTimer {
    public:
        CpuStageTimer(VulkanGSPipeline* pipeline, std::string name)
            : pipeline_(pipeline),
              name_(std::move(name)),
              start_(std::chrono::high_resolution_clock::now()) {}

        CpuStageTimer(const CpuStageTimer&) = delete;
        CpuStageTimer& operator=(const CpuStageTimer&) = delete;
        CpuStageTimer(CpuStageTimer&& other) noexcept
            : pipeline_(std::exchange(other.pipeline_, nullptr)),
              name_(other.name_),
              start_(other.start_) {}
        CpuStageTimer& operator=(CpuStageTimer&&) = delete;

        ~CpuStageTimer() {
            if (!pipeline_ || !pipeline_->cpuTimerCallback_)
                return;
            const auto elapsed = std::chrono::high_resolution_clock::now() - start_;
            const double ms = std::chrono::duration<double, std::milli>(elapsed).count();
            pipeline_->cpuTimerCallback_(name_, ms);
        }

    private:
        VulkanGSPipeline* pipeline_ = nullptr;
        std::string name_;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    };

    CpuStageTimer timeCpuStage(std::string name) {
        return CpuStageTimer(this, std::move(name));
    }

    void bufferMemoryBarrier(const std::vector<std::pair<_VulkanBuffer, BarrierMask>>& buffers, BarrierMask dstMask);

    size_t current_vram = 0;
    size_t peak_vram = 0;

    struct PendingTimelineWait {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        std::uint64_t value = 0;
        VkPipelineStageFlags stage_mask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    };
    std::vector<PendingTimelineWait> pending_timeline_waits_;
    std::unordered_map<VkSemaphore, std::uint64_t> last_timeline_wait_values_;
    std::unordered_map<VkSemaphore, std::uint64_t> last_timeline_signal_values_;

    static constexpr std::uint32_t kCommandBatchSlotCount = 3;
    struct CommandBatchSlot {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkQueryPool timestamp_query_pool = VK_NULL_HANDLE;
        VkSemaphore pending_signal = VK_NULL_HANDLE;
        std::uint64_t pending_signal_value = 0;
        std::uint32_t pending_timestamp_count = 0;
        std::vector<std::pair<int, int>> pending_timestamp_marks;
    };
    std::array<CommandBatchSlot, kCommandBatchSlotCount> command_batch_slots_{};
    std::uint32_t next_command_batch_slot_ = 0;
    std::uint32_t active_command_batch_slot_ = 0;

    // Vulkan objects
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue command_queue;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkQueryPool timestamp_query_pool;
    VmaAllocator allocator = VK_NULL_HANDLE;
    // Persisted, on-disk pipeline cache owned by the host VulkanContext. Optional —
    // VK_NULL_HANDLE simply skips the cache. Shared with the rest of the app's pipelines.
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    PFN_vkCmdPushDescriptorSetKHR vk_cmd_push_descriptor_set_ = nullptr;
    lfs::rendering::VulkanDebugNameWriter debug_name_writer_;

    struct DeviceInfo {
        uint32_t subgroupSize;
        uint32_t sharedSize;
        uint32_t maxGroupsX;
        uint32_t maxGroupsY;
        uint32_t maxGroupsZ;
        uint32_t maxThreadsX;
        uint32_t maxThreadsY;
        uint32_t maxThreadsZ;
    } deviceInfo;

    // Compute pipeline. Storage-buffer bindings are pushed via
    // vkCmdPushDescriptorSetKHR each dispatch — no descriptor pool, no
    // pre-allocated descriptor set, no per-pipeline buffer cache.
    struct _ComputePipeline {
        VkShaderModule shader;
        VkDescriptorSetLayout descriptor_set_layout;
        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;
        std::vector<int> buffer_layouts;
        std::string diagnostic_name;

        _ComputePipeline(
            std::vector<int> buffer_layouts) : shader(VK_NULL_HANDLE),
                                               descriptor_set_layout(VK_NULL_HANDLE),
                                               pipeline_layout(VK_NULL_HANDLE),
                                               pipeline(VK_NULL_HANDLE),
                                               buffer_layouts(buffer_layouts) {}

        _ComputePipeline(int num_buffers)
            : shader(VK_NULL_HANDLE),
              descriptor_set_layout(VK_NULL_HANDLE),
              pipeline_layout(VK_NULL_HANDLE),
              pipeline(VK_NULL_HANDLE) {
            buffer_layouts.resize(num_buffers);
            for (int i = 0; i < num_buffers; i++)
                buffer_layouts[i] = i;
        }
    };

    struct _ComputePipelinePair {
        _ComputePipeline _cp0, _cp1;
        _ComputePipelinePair(int num_buffers)
            : _cp0(num_buffers),
              _cp1(num_buffers) {}
        _ComputePipeline& operator[](bool b) { return b ? _cp0 : _cp1; }
    };

    std::vector<_ComputePipeline*> all_compute_pipelines;

    uint32_t queue_family_index;

    // For CPU-GPU transfers
    struct _Stager {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        size_t allocSize = 0;
        std::mutex mutex;
    };
    _Stager stager;

    void allocStagingBuffer(size_t size);

    void populateDeviceInfo(VkPhysicalDevice selected_physical_device);
    void createCommandPool();
    void createFence();
    void createQueryPools();
    void waitForPendingBatchSlot(CommandBatchSlot& slot);
    void collectTimestampResults(CommandBatchSlot& slot, std::uint32_t timestamp_count);
    void createShaderModule(const std::vector<uint32_t>& spirv_code, VkShaderModule* pShaderModule);
    void validateBufferRange(const _VulkanBuffer& buffer,
                             VkDeviceSize relative_offset,
                             VkDeviceSize size,
                             std::string_view operation) const;
    void validateFillRange(const _VulkanBuffer& buffer,
                           VkDeviceSize relative_offset,
                           VkDeviceSize size,
                           std::string_view operation) const;
    void setDebugObjectName(VkObjectType type, std::uint64_t handle, std::string_view name) const;

    template <typename VkHandle>
    void setDebugObjectName(const VkObjectType type,
                            const VkHandle handle,
                            const std::string_view name) const {
        setDebugObjectName(type, lfs::rendering::vkHandleValue(handle), name);
    }

    void createComputeDescriptorSetLayout(_ComputePipeline& pipeline);
    void createComputePipeline(_ComputePipeline& pipeline, const std::string& spirv_path, uint32_t min_shared_memory = 0, bool compatible_subgroup_size = true);
    void executeCompute(
        std::vector<std::pair<size_t, size_t>> dims,
        const void* uniformsPtr, size_t uniformSize,
        _ComputePipeline& pipeline,
        const std::vector<_VulkanBuffer>& buffers);

    // Indirect dispatch variant. The dispatch group counts come from a
    // GPU-resident VkDispatchIndirectCommand at (indirect_buffer, offset).
    void executeComputeIndirect(
        const _VulkanBuffer& indirect_buffer,
        VkDeviceSize indirect_offset,
        const void* uniformsPtr, size_t uniformSize,
        _ComputePipeline& pipeline,
        const std::vector<_VulkanBuffer>& buffers);

private:
    void destroyComputePipeline(_ComputePipeline& pipeline);
};

class [[nodiscard]] DeviceGuard {
    VulkanGSPipeline* pipeline;
    bool cbip;
    bool use_fence = true;
    VkSemaphore signal_semaphore = VK_NULL_HANDLE;
    std::uint64_t signal_value = 0;
    VkSemaphore secondary_signal_semaphore = VK_NULL_HANDLE;
    std::uint64_t secondary_signal_value = 0;
    const char* debugInfo1 = nullptr;
    int debugInfo2 = -1;
    int uncaught_exceptions = 0;

public:
    DeviceGuard(VulkanGSPipeline* pipeline, const char* debugInfo1 = nullptr, const int debugInfo2 = -1) {
        this->pipeline = pipeline;
        uncaught_exceptions = std::uncaught_exceptions();
        cbip = pipeline->isCommandBatchInProgress();
        if (!cbip) {
            pipeline->beginCommandBatch();
            if (debugInfo1) {
                this->debugInfo1 = debugInfo1;
                this->debugInfo2 = debugInfo2;
                printf("DeviceGuard created: %s:%d\n", debugInfo1, debugInfo2);
            }
        }
    }
    DeviceGuard(VulkanGSPipeline* pipeline,
                const bool use_fence,
                const VkSemaphore signal_semaphore,
                const std::uint64_t signal_value,
                const VkSemaphore secondary_signal_semaphore = VK_NULL_HANDLE,
                const std::uint64_t secondary_signal_value = 0,
                const char* debugInfo1 = nullptr,
                const int debugInfo2 = -1)
        : DeviceGuard(pipeline, debugInfo1, debugInfo2) {
        this->use_fence = use_fence;
        this->signal_semaphore = signal_semaphore;
        this->signal_value = signal_value;
        this->secondary_signal_semaphore = secondary_signal_semaphore;
        this->secondary_signal_value = secondary_signal_value;
    }
    ~DeviceGuard() noexcept(false) {
        if (std::uncaught_exceptions() > uncaught_exceptions) {
            pipeline->cancelCommandBatch();
            return;
        }
        if (!cbip) {
            pipeline->endCommandBatch(use_fence,
                                      signal_semaphore,
                                      signal_value,
                                      secondary_signal_semaphore,
                                      secondary_signal_value);
            if (debugInfo1) {
                printf("DeviceGuard freed: %s:%d\n", debugInfo1, debugInfo2);
            }
        } else if (cbip != pipeline->isCommandBatchInProgress()) {
            _THROW_ERROR(std::format(
                "DeviceGuard batch lifecycle changed unexpectedly (batch_was_active={}, batch_is_active={}, guard_started_batch={})",
                cbip,
                pipeline->isCommandBatchInProgress(),
                !cbip));
        }
    }
};

class [[nodiscard]] HostGuard {
    VulkanGSPipeline* pipeline;
    bool cbip;
    const char* debugInfo1 = nullptr;
    int debugInfo2 = -1;
    int uncaught_exceptions = 0;

public:
    HostGuard(VulkanGSPipeline* pipeline, const char* debugInfo1 = nullptr, const int debugInfo2 = -1) {
        this->pipeline = pipeline;
        uncaught_exceptions = std::uncaught_exceptions();
        cbip = pipeline->isCommandBatchInProgress();
        if (cbip) {
            pipeline->endCommandBatch();
            if (debugInfo1) {
                this->debugInfo1 = debugInfo1;
                this->debugInfo2 = debugInfo2;
                printf("HostGuard created: %s:%d\n", debugInfo1, debugInfo2);
            }
        }
    }
    ~HostGuard() noexcept(false) {
        if (std::uncaught_exceptions() > uncaught_exceptions) {
            pipeline->cancelCommandBatch();
            return;
        }
        if (cbip) {
            pipeline->beginCommandBatch();
            if (debugInfo1) {
                printf("HostGuard freed: %s:%d\n", debugInfo1, debugInfo2);
            }
        } else if (cbip != pipeline->isCommandBatchInProgress()) {
            pipeline->cancelCommandBatch();
            _THROW_ERROR(std::format(
                "HostGuard batch lifecycle changed unexpectedly (batch_was_active={}, batch_is_active={}, guard_paused_batch={})",
                cbip,
                pipeline->isCommandBatchInProgress(),
                cbip));
        }
    }
};

#define DEVICE_GUARD auto deviceGuard = DeviceGuard(this)
#define HOST_GUARD   auto hostGuard = HostGuard(this)
