#pragma once

#include <algorithm> // std::sort
#include <array>
#include <cstring> // memcpy
#include <functional>
#include <map>
#include <string>
#include <variant>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cassert>

#include "buffer.h"

union Uniform32_t {
    uint32_t u;
    float f;
};

class VulkanGSPipeline {
public:
    VulkanGSPipeline();
    ~VulkanGSPipeline();

    void initializeExternal(VkInstance external_instance,
                            VkPhysicalDevice external_physical_device,
                            VkDevice external_device,
                            VkQueue external_queue,
                            uint32_t external_queue_family_index,
                            VmaAllocator external_allocator);
    void cleanup();
    void cleanupBuffers(VulkanGSPipelineBuffers& buffers);

    void createBuffer(size_t size, _VulkanBuffer& buffer);
    void destroyBuffer(_VulkanBuffer& buffer);
    void resizeDeviceBuffer(_VulkanBuffer& deviceBuffer, size_t new_byte_size, bool no_shrink = true);
    template <typename T>
    _VulkanBuffer& resizeDeviceBuffer(Buffer<T>& buffer, size_t new_size, bool no_shrink = true);
    template <typename T>
    _VulkanBuffer& clearDeviceBuffer(Buffer<T>& buffer, size_t new_size);
    template <typename T>
    _VulkanBuffer& resizeAndCopyDeviceBuffer(Buffer<T>& buffer, size_t new_size, bool clear);
    template <typename T>
    T readElement(const _VulkanBuffer& buffer, size_t index);

    void beginCommandBatch();
    void endCommandBatch(bool use_fence = true);
    bool isCommandBatchInProgress() const {
        return commandBatchInProgress;
    }
    VkCommandBuffer activeCommandBuffer() const {
        return command_buffer;
    }
    bool writeTimestamp(int delta);
    bool writeTimestampNoExcept(int delta);

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
    std::vector<std::function<void(const std::vector<std::pair<size_t, double>>&)>> timerCallbacks;

    void bufferMemoryBarrier(const std::vector<std::pair<_VulkanBuffer, BarrierMask>>& buffers, BarrierMask dstMask);

    size_t current_vram = 0;
    size_t peak_vram = 0;

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
    PFN_vkCmdPushDescriptorSetKHR vk_cmd_push_descriptor_set_ = nullptr;

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
    void createShaderModule(const std::vector<uint32_t>& spirv_code, VkShaderModule* pShaderModule);

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
    const char* debugInfo1 = nullptr;
    int debugInfo2 = -1;

public:
    DeviceGuard(VulkanGSPipeline* pipeline, const char* debugInfo1 = nullptr, const int debugInfo2 = -1) {
        // printf("DeviceGuard constructor\n");
        this->pipeline = pipeline;
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
    ~DeviceGuard() noexcept(false) {
        // printf("DeviceGuard destructor\n");
        if (!cbip) {
            pipeline->endCommandBatch();
            if (debugInfo1) {
                printf("DeviceGuard freed: %s:%d\n", debugInfo1, debugInfo2);
            }
        } else if (cbip != pipeline->isCommandBatchInProgress()) {
            fprintf(stderr, "commandBatchInProgress changed during DeviceGuard (originally %d)\n", (int)cbip);
            std::terminate();
        }
    }
};

class [[nodiscard]] HostGuard {
    VulkanGSPipeline* pipeline;
    bool cbip;
    const char* debugInfo1 = nullptr;
    int debugInfo2 = -1;

public:
    HostGuard(VulkanGSPipeline* pipeline, const char* debugInfo1 = nullptr, const int debugInfo2 = -1) {
        // printf("HostGuard constructor\n");
        this->pipeline = pipeline;
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
        // printf("HostGuard destructor\n");
        if (cbip) {
            pipeline->beginCommandBatch();
            if (debugInfo1) {
                printf("HostGuard freed: %s:%d\n", debugInfo1, debugInfo2);
            }
        } else if (cbip != pipeline->isCommandBatchInProgress()) {
            fprintf(stderr, "commandBatchInProgress changed during HostGuard (originally %d)\n", (int)cbip);
            std::terminate();
        }
    }
};

// #define DeviceGuard(args) DeviceGuard(args, __FILE__, __LINE__)
// #define HostGuard(args) HostGuard(args, __FILE__, __LINE__)

#define DEVICE_GUARD auto deviceGuard = DeviceGuard(this)
#define HOST_GUARD   auto hostGuard = HostGuard(this)
