#include "gs_pipeline.h"
#include "perf_timer.h"

#include <fstream>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

static const size_t MAX_UNIFORM_SIZE = 192;

static const uint32_t MAX_TIMESTAMP_QUERY_COUNT = 48;

#if defined(__SSE2__) || defined(_MSC_VER)
#define SSE2_AVAILABLE 1
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

std::vector<uint32_t> loadSpirv(std::string spirv_path) {
// Load the SPIR-V file
#ifdef WIN32
    // replace "/" with "\\"
    size_t start_pos = 0;
    while ((start_pos = spirv_path.find("/", start_pos)) != std::string::npos) {
        spirv_path.replace(start_pos, 1, "\\");
        start_pos += 1;
    }
#endif

    std::ifstream file(spirv_path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Failed to open file: " + spirv_path);

    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> spirv_code(fileSize / sizeof(uint32_t));
    if (!file.read(reinterpret_cast<char*>(spirv_code.data()), fileSize))
        throw std::runtime_error("Failed to read file: " + spirv_path);

    return spirv_code;
}

VulkanGSPipeline::VulkanGSPipeline() : instance(VK_NULL_HANDLE),
                                       physical_device(VK_NULL_HANDLE),
                                       device(VK_NULL_HANDLE),
                                       command_queue(VK_NULL_HANDLE),
                                       command_pool(VK_NULL_HANDLE),
                                       command_buffer(VK_NULL_HANDLE),
                                       fence(VK_NULL_HANDLE),
                                       timestamp_query_pool(VK_NULL_HANDLE),
                                       queue_family_index(UINT32_MAX) {
}

VulkanGSPipeline::~VulkanGSPipeline() {
    if (commandBatchInProgress)
        endCommandBatch(false);
    cleanup();
}

void VulkanGSPipeline::initializeExternal(VkInstance external_instance,
                                          VkPhysicalDevice external_physical_device,
                                          VkDevice external_device,
                                          VkQueue external_queue,
                                          uint32_t external_queue_family_index,
                                          VmaAllocator external_allocator) {
    cleanup();
    if (external_instance == VK_NULL_HANDLE ||
        external_physical_device == VK_NULL_HANDLE ||
        external_device == VK_NULL_HANDLE ||
        external_queue == VK_NULL_HANDLE ||
        external_queue_family_index == UINT32_MAX ||
        external_allocator == VK_NULL_HANDLE) {
        _THROW_ERROR("initializeExternal received an invalid Vulkan handle");
    }

    instance = external_instance;
    physical_device = external_physical_device;
    device = external_device;
    command_queue = external_queue;
    queue_family_index = external_queue_family_index;
    allocator = external_allocator;

    vk_cmd_push_descriptor_set_ = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
        vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR"));
    if (vk_cmd_push_descriptor_set_ == nullptr) {
        _THROW_ERROR("VK_KHR_push_descriptor is required by vksplat compute pipeline but not available on this device");
    }

    populateDeviceInfo(physical_device);
    createCommandPool();
    createFence();
    createQueryPools();

    commandBatchInProgress = false;
}

void VulkanGSPipeline::cleanupBuffers(VulkanGSPipelineBuffers& buffers) {
    HOST_GUARD;
#define _(name)                                   \
    {                                             \
        destroyBuffer(buffers.name.deviceBuffer); \
        buffers.name.clear();                     \
        buffers.name.shrink_to_fit();             \
    }
    _(xyz_ws)
    _(sh_coeffs)
    _(rotations)
    _(scales_opacs)
    _(tiles_touched)
    _(rect_tile_space)
    _(radii)
    _(xy_vs)
    _(depths)
    _(inv_cov_vs_opacity)
    _(rgb)
    _(overlay_flags)
    _(index_buffer_offset)
    _(sorting_keys_1)
    _(sorting_keys_2)
    _(sorting_gauss_idx_1)
    _(sorting_gauss_idx_2)
    _(tile_ranges)
    _(pixel_state)
    _(pixel_depth)
    _(n_contributors)
    _(_cumsum_blockSums)
    _(_cumsum_blockSums2)
    _(_sorting_histogram)
    _(_sorting_histogram_cumsum)
#undef _
}

void VulkanGSPipeline::cleanup() {
    // Pipeline never owns the Vulkan instance, device, or VMA allocator —
    // those are always passed in by the host visualizer via initializeExternal.
    // Clean up only what we created on top of them.
    HOST_GUARD;

    if (stager.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, stager.buffer, stager.allocation);
        stager.buffer = VK_NULL_HANDLE;
        stager.allocation = VK_NULL_HANDLE;
        stager.allocSize = 0;
    }

    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);

        for (_ComputePipeline* pipeline : all_compute_pipelines)
            destroyComputePipeline(*pipeline);
        all_compute_pipelines.clear();

        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, fence, nullptr);
            fence = VK_NULL_HANDLE;
        }
        if (timestamp_query_pool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, timestamp_query_pool, nullptr);
            timestamp_query_pool = VK_NULL_HANDLE;
        }

        if (command_buffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
            command_buffer = VK_NULL_HANDLE;
        }
        if (command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, command_pool, nullptr);
            command_pool = VK_NULL_HANDLE;
        }
    }

    allocator = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    instance = VK_NULL_HANDLE;
    physical_device = VK_NULL_HANDLE;
    command_queue = VK_NULL_HANDLE;
    queue_family_index = UINT32_MAX;
}

void VulkanGSPipeline::populateDeviceInfo(VkPhysicalDevice selected_physical_device) {
    VkPhysicalDeviceSubgroupProperties subgroupProperties{};
    subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 deviceProperties2{};
    deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProperties2.pNext = &subgroupProperties;
    vkGetPhysicalDeviceProperties2(selected_physical_device, &deviceProperties2);
    const auto& limits = deviceProperties2.properties.limits;

    deviceInfo = {
        subgroupProperties.subgroupSize,
        limits.maxComputeSharedMemorySize,
        limits.maxComputeWorkGroupCount[0],
        limits.maxComputeWorkGroupCount[1],
        limits.maxComputeWorkGroupCount[2],
        limits.maxComputeWorkGroupSize[0],
        limits.maxComputeWorkGroupSize[1],
        limits.maxComputeWorkGroupSize[2],
    };
}

void VulkanGSPipeline::createCommandPool() {
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family_index;

    if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS)
        _THROW_ERROR("Failed to create command pool");

    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool;
    alloc_info.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &alloc_info, &command_buffer) != VK_SUCCESS)
        _THROW_ERROR("Failed to allocate command buffer");
}

void VulkanGSPipeline::createFence() {
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;

    VkResult result = vkCreateFence(device, &fenceInfo, nullptr, &fence);
    if (result != VK_SUCCESS)
        _THROW_ERROR("Failed to create fence");
}

void VulkanGSPipeline::createQueryPools() {
    // timestamp
    VkQueryPoolCreateInfo queryPoolCreateInfo = {};
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = MAX_TIMESTAMP_QUERY_COUNT;
    if (vkCreateQueryPool(device, &queryPoolCreateInfo, nullptr, &timestamp_query_pool) != VK_SUCCESS)
        _THROW_ERROR("Failed to create timestamp query pool");
}

void VulkanGSPipeline::createShaderModule(const std::vector<uint32_t>& spirv_code, VkShaderModule* pShaderModule) {
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_code.size() * sizeof(uint32_t);
    create_info.pCode = spirv_code.data();

    if (vkCreateShaderModule(device, &create_info, nullptr, pShaderModule) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module");
}

void VulkanGSPipeline::beginCommandBatch() {
    if (commandBatchInProgress)
        _THROW_ERROR("Command batch already in progress");
    commandBatchInProgress = true;

    PerfTimer::hostToc();

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS)
        _THROW_ERROR("Failed to begin command buffer for batch");

    vkCmdResetQueryPool(command_buffer, timestamp_query_pool, 0, MAX_TIMESTAMP_QUERY_COUNT);
    PerfTimer::popMarkers(this);
}

void VulkanGSPipeline::endCommandBatch(bool use_fence) {
    if (!commandBatchInProgress)
        _THROW_ERROR("No command batch in progress");

    if (timestampNumWritten > 0) {
        while (timestampStackDepth > 0)
            PerfTimer::pushMarker(this);
    }

    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        _THROW_ERROR("Failed to end command buffer for batch");
    }

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    if (vkQueueSubmit(command_queue, 1, &submit_info,
                      use_fence ? fence : VK_NULL_HANDLE) != VK_SUCCESS) {
        _THROW_ERROR("Failed to submit batch");
    }

    commandBatchInProgress = false;

    if (use_fence) {
#if SSE2_AVAILABLE
#if ENABLE_ASSERTION
        constexpr unsigned long long kTimeout = 0x100000000ull;
        auto time0 = __rdtsc();
#endif
        while (vkGetFenceStatus(device, fence) != VK_SUCCESS) {
            _mm_pause();
#if ENABLE_ASSERTION
            if (__rdtsc() - time0 >= kTimeout) {
                // _THROW_ERROR("Fence timed out");
                printf("\033[91m%s\033[m\n", "Timed out.");
                std::terminate(); // note that this is often in destructor
            }
#endif
        }
#else
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
#endif
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS)
            _THROW_ERROR("Failed to reset fence");
    } else
        vkQueueWaitIdle(command_queue);

    PerfTimer::hostTic();

    if (timestampNumWritten > 0) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(physical_device, &deviceProperties);
        double timestampPeriod = deviceProperties.limits.timestampPeriod;

        std::vector<uint64_t> timestamps(timestampNumWritten);
        vkGetQueryPoolResults(
            device, timestamp_query_pool,
            0, timestampNumWritten,
            sizeof(uint64_t) * timestampNumWritten,
            timestamps.data(), sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        std::vector<double> times(timestampNumWritten);
        for (uint32_t i = 0; i < timestampNumWritten; i++)
            times[i] = 1e-9 * double(timestamps[i] - timestamps[0]) * timestampPeriod;
        auto time_updates = PerfTimer::update(times);
        for (auto& callback : timerCallbacks)
            callback(time_updates);

        timestampNumWritten = 0;
    }
}

bool VulkanGSPipeline::writeTimestamp(int delta) {
    if (!commandBatchInProgress)
        _THROW_ERROR("writeTimestamp requires command batch in progress");
    if (timestampNumWritten >= MAX_TIMESTAMP_QUERY_COUNT)
        _THROW_ERROR("Too many timestamps written");
    if (delta != 1 && delta != -1)
        _THROW_ERROR("delta in writeTimestamp must be 1 or -1");
    if (delta == -1 && timestampStackDepth == 0)
        _THROW_ERROR("attempt to write exit timestamp while stack is empty");
    vkCmdWriteTimestamp(
        command_buffer,
        // delta == 1 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        timestamp_query_pool, timestampNumWritten);
    timestampNumWritten += 1;
    timestampStackDepth += delta;
    return true;
}

bool VulkanGSPipeline::writeTimestampNoExcept(int delta) {
    if (!commandBatchInProgress)
        return false;
    if (timestampNumWritten >= MAX_TIMESTAMP_QUERY_COUNT)
        return false;
    if (delta != 1 && delta != -1)
        return false;
    if (delta == -1 && timestampStackDepth == 0)
        return false;
    vkCmdWriteTimestamp(
        command_buffer,
        // delta == 1 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        timestamp_query_pool, timestampNumWritten);
    timestampNumWritten += 1;
    timestampStackDepth += delta;
    return true;
}

VkAccessFlags toAccessMask(VulkanGSPipeline::BarrierMask barrierMask) {
    VkAccessFlags result = (VkAccessFlags)0;
    if (barrierMask == VulkanGSPipeline::TRANSFER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_ACCESS_TRANSFER_READ_BIT;
    if (barrierMask == VulkanGSPipeline::TRANSFER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_ACCESS_TRANSFER_WRITE_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_ACCESS_SHADER_READ_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_ACCESS_SHADER_WRITE_BIT;
    if (barrierMask == VulkanGSPipeline::INDIRECT_DISPATCH_READ)
        result |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    return result;
}

VkPipelineStageFlags toStageMask(VulkanGSPipeline::BarrierMask barrierMask) {
    VkPipelineStageFlags result = (VkPipelineStageFlags)0;
    if (barrierMask == VulkanGSPipeline::TRANSFER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (barrierMask == VulkanGSPipeline::HOST_READ ||
        barrierMask == VulkanGSPipeline::HOST_WRITE ||
        barrierMask == VulkanGSPipeline::HOST_READ_WRITE)
        result |= VK_PIPELINE_STAGE_HOST_BIT;
    if (barrierMask == VulkanGSPipeline::INDIRECT_DISPATCH_READ)
        result |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    return result;
}

void VulkanGSPipeline::bufferMemoryBarrier(
    const std::vector<std::pair<_VulkanBuffer, VulkanGSPipeline::BarrierMask>>& buffers,
    VulkanGSPipeline::BarrierMask dstMask) {
    if (!commandBatchInProgress)
        return;

    std::vector<VkBufferMemoryBarrier> barriers;
    barriers.reserve(buffers.size());
    VkPipelineStageFlags srcStageFlags = (VkPipelineStageFlags)0;
    for (auto& [buffer, srcMask] : buffers) {
        if (buffer.buffer == VK_NULL_HANDLE)
            continue;
        VkBufferMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask = toAccessMask(srcMask);
        barrier.dstAccessMask = toAccessMask(dstMask);
        barrier.srcQueueFamilyIndex = queue_family_index;
        barrier.dstQueueFamilyIndex = queue_family_index;
        barrier.buffer = buffer.buffer;
        barrier.offset = 0;
        barrier.size = buffer.size;
        barriers.push_back(barrier);
        srcStageFlags |= toStageMask(srcMask);
    }
    if (barriers.empty())
        return;

    vkCmdPipelineBarrier(
        command_buffer,
        srcStageFlags, toStageMask(dstMask),
        0,                                          // dependencyFlags
        0, nullptr,                                 // memory barriers
        (uint32_t)barriers.size(), barriers.data(), // buffer barriers
        0, nullptr                                  // image barriers
    );
}

// Compute pipeline

void VulkanGSPipeline::createComputeDescriptorSetLayout(_ComputePipeline& pipeline) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(pipeline.buffer_layouts.size());

    for (int i : pipeline.buffer_layouts) {
        VkDescriptorSetLayoutBinding binding;
        binding.binding = i;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binding.pImmutableSamplers = nullptr;
        bindings.push_back(binding);
    }

    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &pipeline.descriptor_set_layout) != VK_SUCCESS)
        _THROW_ERROR("Failed to create descriptor set layout");
}

void VulkanGSPipeline::createComputePipeline(_ComputePipeline& pipeline, const std::string& spirv_path, uint32_t min_shared_memory, bool compatible_subgroup_size) {

    if (min_shared_memory > this->deviceInfo.sharedSize) {
        pipeline.shader = VK_NULL_HANDLE;
        return;
    }

    createShaderModule(loadSpirv(spirv_path), &pipeline.shader);
    createComputeDescriptorSetLayout(pipeline);

    // Create push constant range for uniforms
    VkPushConstantRange push_constant_range = {};
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = (uint32_t)MAX_UNIFORM_SIZE;

    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &pipeline.descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant_range;

    if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline.pipeline_layout) != VK_SUCCESS) {
        _THROW_ERROR("Failed to create pipeline set layout");
    }

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT req = {};
    req.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
    req.requiredSubgroupSize = SUBGROUP_SIZE; // 32

    VkPipelineShaderStageCreateInfo compute_shader_stage_info = {};
    compute_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute_shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute_shader_stage_info.module = pipeline.shader;
    compute_shader_stage_info.pName = "main";
    if (compatible_subgroup_size && deviceInfo.subgroupSize != SUBGROUP_SIZE)
        compute_shader_stage_info.pNext = &req;

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = pipeline.pipeline_layout;
    pipeline_info.stage = compute_shader_stage_info;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline.pipeline) != VK_SUCCESS)
        _THROW_ERROR("Failed to create compute pipeline");

    all_compute_pipelines.push_back(&pipeline);
}

void VulkanGSPipeline::executeCompute(
    std::vector<std::pair<size_t, size_t>> dims,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline& pipeline,
    const std::vector<_VulkanBuffer>& buffers) {
    if (uniformSize > MAX_UNIFORM_SIZE)
        _THROW_ERROR("Maximum uniform size exceeded");

    DEVICE_GUARD;

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

    const std::size_t num_buffers = pipeline.buffer_layouts.size();
    std::vector<VkDescriptorBufferInfo> buffer_infos(num_buffers);
    std::vector<VkWriteDescriptorSet> writes(num_buffers);
    for (std::size_t idx = 0; idx < num_buffers; ++idx) {
        const int binding = pipeline.buffer_layouts[idx];
        if (buffers[binding].buffer == VK_NULL_HANDLE)
            _THROW_ERROR("Buffer " + std::to_string(binding) + " is NULL");
        buffer_infos[idx].buffer = buffers[binding].buffer;
        buffer_infos[idx].offset = buffers[binding].offset;
        // Bind the in-use [offset, offset+size) range. For owned buffers size
        // is set by resizeDeviceBuffer / createBuffer to match the requested
        // allocation; for coalesced views into a parent allocation it's the
        // sub-region's payload byte count. Falling back to allocSize when size
        // is zero keeps any (rare) legacy callers working without surprises.
        buffer_infos[idx].range = buffers[binding].size != 0
                                      ? buffers[binding].size
                                      : buffers[binding].allocSize;

        writes[idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[idx].dstSet = VK_NULL_HANDLE; // ignored for push descriptor
        writes[idx].dstBinding = static_cast<uint32_t>(binding);
        writes[idx].dstArrayElement = 0;
        writes[idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[idx].descriptorCount = 1;
        writes[idx].pBufferInfo = &buffer_infos[idx];
    }
    vk_cmd_push_descriptor_set_(command_buffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline.pipeline_layout,
                                0,
                                static_cast<uint32_t>(writes.size()),
                                writes.data());

    // Push constants for uniforms
    if (uniformsPtr) {
        vkCmdPushConstants(
            command_buffer,
            pipeline.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, (uint32_t)uniformSize, uniformsPtr);
    }

    // Dispatch compute shader
    while (dims.size() < 3)
        dims.push_back({1, 1});
    uint32_t nGroupsX = (uint32_t)_CEIL_DIV(dims[0].first, dims[0].second);
    uint32_t nGroupsY = (uint32_t)_CEIL_DIV(dims[1].first, dims[1].second);
    uint32_t nGroupsZ = (uint32_t)_CEIL_DIV(dims[2].first, dims[2].second);
    if (nGroupsX > deviceInfo.maxGroupsX ||
        nGroupsY > deviceInfo.maxGroupsY ||
        nGroupsZ > deviceInfo.maxGroupsZ)
        _THROW_ERROR("Cannot launch compute kernel, too many groups: [" +
                     std::to_string(nGroupsX) + " " +
                     std::to_string(nGroupsY) + " " +
                     std::to_string(nGroupsZ) + "] > [" +
                     std::to_string(deviceInfo.maxGroupsX) + " " +
                     std::to_string(deviceInfo.maxGroupsY) + " " +
                     std::to_string(deviceInfo.maxGroupsZ) + "]");
    vkCmdDispatch(command_buffer, nGroupsX, nGroupsY, nGroupsZ);
}

void VulkanGSPipeline::executeComputeIndirect(
    const _VulkanBuffer& indirect_buffer,
    VkDeviceSize indirect_offset,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline& pipeline,
    const std::vector<_VulkanBuffer>& buffers) {
    if (uniformSize > MAX_UNIFORM_SIZE)
        _THROW_ERROR("Maximum uniform size exceeded");
    if (indirect_buffer.buffer == VK_NULL_HANDLE)
        _THROW_ERROR("Indirect dispatch buffer is NULL");

    DEVICE_GUARD;

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

    const std::size_t num_buffers = pipeline.buffer_layouts.size();
    std::vector<VkDescriptorBufferInfo> buffer_infos(num_buffers);
    std::vector<VkWriteDescriptorSet> writes(num_buffers);
    for (std::size_t idx = 0; idx < num_buffers; ++idx) {
        const int binding = pipeline.buffer_layouts[idx];
        if (buffers[binding].buffer == VK_NULL_HANDLE)
            _THROW_ERROR("Buffer " + std::to_string(binding) + " is NULL");
        buffer_infos[idx].buffer = buffers[binding].buffer;
        buffer_infos[idx].offset = buffers[binding].offset;
        // Bind the in-use [offset, offset+size) range. For owned buffers size
        // is set by resizeDeviceBuffer / createBuffer to match the requested
        // allocation; for coalesced views into a parent allocation it's the
        // sub-region's payload byte count. Falling back to allocSize when size
        // is zero keeps any (rare) legacy callers working without surprises.
        buffer_infos[idx].range = buffers[binding].size != 0
                                      ? buffers[binding].size
                                      : buffers[binding].allocSize;

        writes[idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[idx].dstSet = VK_NULL_HANDLE;
        writes[idx].dstBinding = static_cast<uint32_t>(binding);
        writes[idx].dstArrayElement = 0;
        writes[idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[idx].descriptorCount = 1;
        writes[idx].pBufferInfo = &buffer_infos[idx];
    }
    vk_cmd_push_descriptor_set_(command_buffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline.pipeline_layout,
                                0,
                                static_cast<uint32_t>(writes.size()),
                                writes.data());

    if (uniformsPtr) {
        vkCmdPushConstants(
            command_buffer,
            pipeline.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, (uint32_t)uniformSize, uniformsPtr);
    }

    vkCmdDispatchIndirect(command_buffer, indirect_buffer.buffer, indirect_offset);
}

void VulkanGSPipeline::destroyComputePipeline(_ComputePipeline& pipeline) {
    if (pipeline.descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, pipeline.descriptor_set_layout, nullptr);
        pipeline.descriptor_set_layout = VK_NULL_HANDLE;
    }
    if (pipeline.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipeline.pipeline, nullptr);
        pipeline.pipeline = VK_NULL_HANDLE;
    }
    if (pipeline.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipeline.pipeline_layout, nullptr);
        pipeline.pipeline_layout = VK_NULL_HANDLE;
    }
    if (pipeline.shader != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, pipeline.shader, nullptr);
        pipeline.shader = VK_NULL_HANDLE;
    }
}
