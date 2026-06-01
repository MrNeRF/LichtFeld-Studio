#include "gs_pipeline.h"
#include "perf_timer.h"

#include "diagnostics/vram_profiler.hpp"

#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

static const size_t MAX_UNIFORM_SIZE = 192;

static const uint32_t MAX_TIMESTAMP_QUERY_COUNT = 96;

namespace {
    constexpr std::string_view kSlangShaderBytecodeScope = "vksplat.shaders.slang.spirv";
    constexpr std::string_view kSlangShaderRootScope = "vksplat.shaders.slang";

    [[nodiscard]] bool isGeneratedSlangSpirvPath(const std::string& spirv_path) {
        return spirv_path.find("/generated/") != std::string::npos ||
               spirv_path.find("\\generated\\") != std::string::npos;
    }

    [[nodiscard]] std::string spirvDiagnosticName(const std::string& spirv_path) {
        auto name = std::filesystem::path(spirv_path).stem().string();
        if (name.empty()) {
            return "unnamed";
        }
        if (name == "projection_forward") {
            return "projection_forward/standard";
        }
        if (name == "projection_forward_3dgut") {
            return "projection_forward/3dgut";
        }
        if (name == "rasterize_forward") {
            return "rasterize_forward/standard";
        }
        if (name == "rasterize_forward_3dgut") {
            return "rasterize_forward/3dgut";
        }
        if (name == "rasterize_forward_plain") {
            return "rasterize_forward/standard_plain";
        }
        if (name == "rasterize_forward_3dgut_plain") {
            return "rasterize_forward/3dgut_plain";
        }
        constexpr std::string_view cumsum_prefix = "cumsum_";
        if (name.rfind(cumsum_prefix, 0) == 0) {
            return "cumsum/" + name.substr(cumsum_prefix.size());
        }
        return name;
    }

    void recordSlangShaderBytecode(const std::string& spirv_path, const std::size_t bytes) {
        if (!isGeneratedSlangSpirvPath(spirv_path) || bytes == 0) {
            return;
        }
        lfs::diagnostics::VramProfiler::instance().recordStaticBytes(
            kSlangShaderBytecodeScope,
            spirvDiagnosticName(spirv_path),
            bytes);
    }
} // namespace

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
                                          VmaAllocator external_allocator,
                                          VkPipelineCache external_pipeline_cache) {
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
    pipeline_cache = external_pipeline_cache;

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

void VulkanGSPipeline::assignBufferLabels(VulkanGSPipelineBuffers& buffers) {
    // Assign a diagnostic label to every device-side buffer so the VRAM HUD can
    // split the Vulkan footprint into per-buffer rows. Labels are stable string
    // literals; the address is stored in _VulkanBuffer::label and lives forever.
#define _(name) buffers.name.deviceBuffer.label = #name;
    _(xyz_ws)
    _(sh_coeffs)
    _(rotations)
    _(scales_opacs)
    _(sh0)
    _(shN)
    _(scaling_raw)
    _(opacity_raw)
    _(tiles_touched)
    _(rect_tile_space)
    _(radii)
    _(xy_vs)
    _(depths)
    _(inv_cov_vs_opacity)
    _(rgb)
    _(overlay_flags)
    _(primitive_depth_keys)
    _(primitive_sort_indices)
    _(tiles_touched_depth_ordered)
    _(visible_flags)
    _(visible_prefix)
    _(visible_count)
    _(visible_sort_dispatch_args)
    _(index_buffer_offset)
    _(sorting_keys_1)
    _(sorting_keys_2)
    _(sorting_gauss_idx_1)
    _(sorting_gauss_idx_2)
    _(tile_sort_count)
    _(tile_sort_dispatch_args)
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

void VulkanGSPipeline::cleanupBuffers(VulkanGSPipelineBuffers& buffers) {
    HOST_GUARD;
    waitForPendingBatch();
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
    _(sh0)
    _(shN)
    _(scaling_raw)
    _(opacity_raw)
    _(tiles_touched)
    _(rect_tile_space)
    _(radii)
    _(xy_vs)
    _(depths)
    _(inv_cov_vs_opacity)
    _(rgb)
    _(overlay_flags)
    _(primitive_depth_keys)
    _(primitive_sort_indices)
    _(tiles_touched_depth_ordered)
    _(visible_flags)
    _(visible_prefix)
    _(visible_count)
    _(visible_sort_dispatch_args)
    _(index_buffer_offset)
    _(sorting_keys_1)
    _(sorting_keys_2)
    _(sorting_gauss_idx_1)
    _(sorting_gauss_idx_2)
    _(tile_sort_count)
    _(tile_sort_dispatch_args)
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
    if (!commandBatchInProgress) {
        waitForPendingBatch();
    }
    HOST_GUARD;
    lfs::diagnostics::VramProfiler::instance().clearStaticScope(kSlangShaderRootScope);

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
        for (CommandBatchSlot& slot : command_batch_slots_) {
            if (slot.timestamp_query_pool != VK_NULL_HANDLE) {
                vkDestroyQueryPool(device, slot.timestamp_query_pool, nullptr);
                slot.timestamp_query_pool = VK_NULL_HANDLE;
            }
        }

        std::array<VkCommandBuffer, kCommandBatchSlotCount> command_buffers{};
        std::uint32_t command_buffer_count = 0;
        for (CommandBatchSlot& slot : command_batch_slots_) {
            if (slot.command_buffer != VK_NULL_HANDLE) {
                command_buffers[command_buffer_count++] = slot.command_buffer;
                slot.command_buffer = VK_NULL_HANDLE;
            }
        }
        if (command_buffer_count > 0) {
            vkFreeCommandBuffers(device, command_pool, command_buffer_count, command_buffers.data());
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
    pending_timeline_waits_.clear();
    for (CommandBatchSlot& slot : command_batch_slots_) {
        slot.pending_signal = VK_NULL_HANDLE;
        slot.pending_signal_value = 0;
        slot.pending_timestamp_count = 0;
        slot.pending_timestamp_marks.clear();
    }
    command_buffer = VK_NULL_HANDLE;
    timestamp_query_pool = VK_NULL_HANDLE;
    next_command_batch_slot_ = 0;
    active_command_batch_slot_ = 0;
    timerCallbacks.clear();
    cpuTimerCallback_ = {};
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

    std::array<VkCommandBuffer, kCommandBatchSlotCount> command_buffers{};
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool;
    alloc_info.commandBufferCount = kCommandBatchSlotCount;

    if (vkAllocateCommandBuffers(device, &alloc_info, command_buffers.data()) != VK_SUCCESS)
        _THROW_ERROR("Failed to allocate command buffer");
    for (std::uint32_t i = 0; i < kCommandBatchSlotCount; ++i) {
        command_batch_slots_[i].command_buffer = command_buffers[i];
    }
    command_buffer = command_batch_slots_[0].command_buffer;
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
    for (CommandBatchSlot& slot : command_batch_slots_) {
        if (vkCreateQueryPool(device, &queryPoolCreateInfo, nullptr, &slot.timestamp_query_pool) != VK_SUCCESS)
            _THROW_ERROR("Failed to create timestamp query pool");
    }
    timestamp_query_pool = command_batch_slots_[0].timestamp_query_pool;
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

    active_command_batch_slot_ = next_command_batch_slot_;
    next_command_batch_slot_ = (next_command_batch_slot_ + 1) % kCommandBatchSlotCount;
    CommandBatchSlot& slot = command_batch_slots_[active_command_batch_slot_];
    waitForPendingBatchSlot(slot);
    command_buffer = slot.command_buffer;
    timestamp_query_pool = slot.timestamp_query_pool;
    timestampNumWritten = 0;
    timestampStackDepth = 0;
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

void VulkanGSPipeline::waitForPendingBatch() {
    for (CommandBatchSlot& slot : command_batch_slots_) {
        waitForPendingBatchSlot(slot);
    }
}

bool VulkanGSPipeline::timelineValueComplete(const VkSemaphore semaphore,
                                             const std::uint64_t value) const {
    if (semaphore == VK_NULL_HANDLE || value == 0)
        return true;
    std::uint64_t completed_value = 0;
    const VkResult result = vkGetSemaphoreCounterValue(device, semaphore, &completed_value);
    if (result != VK_SUCCESS)
        _THROW_ERROR("Failed to query timeline semaphore");
    return completed_value >= value;
}

void VulkanGSPipeline::waitForPendingBatchSlot(CommandBatchSlot& slot) {
    if (slot.pending_signal == VK_NULL_HANDLE || slot.pending_signal_value == 0)
        return;

    {
        [[maybe_unused]] auto cpu_timer = timeCpuStage("vksplat.command_batch.wait_pending");
        if (!timelineValueComplete(slot.pending_signal, slot.pending_signal_value)) {
            VkSemaphoreWaitInfo wait_info{};
            wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
            wait_info.semaphoreCount = 1;
            wait_info.pSemaphores = &slot.pending_signal;
            wait_info.pValues = &slot.pending_signal_value;
            const VkResult result = vkWaitSemaphores(device, &wait_info, UINT64_MAX);
            if (result != VK_SUCCESS)
                _THROW_ERROR("Failed to wait for pending batch timeline semaphore");
        }
    }

    collectTimestampResults(slot, slot.pending_timestamp_count);
    slot.pending_signal = VK_NULL_HANDLE;
    slot.pending_signal_value = 0;
    slot.pending_timestamp_count = 0;
    slot.pending_timestamp_marks.clear();
}

void VulkanGSPipeline::collectTimestampResults(CommandBatchSlot& slot,
                                               const std::uint32_t timestamp_count) {
    if (timestamp_count == 0)
        return;
    [[maybe_unused]] auto cpu_timer = timeCpuStage("vksplat.command_batch.query_results");
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physical_device, &deviceProperties);
    double timestampPeriod = deviceProperties.limits.timestampPeriod;

    std::vector<uint64_t> timestamps(timestamp_count);
    vkGetQueryPoolResults(
        device, slot.timestamp_query_pool,
        0, timestamp_count,
        sizeof(uint64_t) * timestamp_count,
        timestamps.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    std::vector<double> times(timestamp_count);
    for (uint32_t i = 0; i < timestamp_count; i++)
        times[i] = 1e-9 * double(timestamps[i] - timestamps[0]) * timestampPeriod;
    auto time_updates = PerfTimer::update(times, slot.pending_timestamp_marks);
    for (auto& callback : timerCallbacks)
        callback(time_updates);
}

void VulkanGSPipeline::addTimelineWait(
    const VkSemaphore semaphore,
    const std::uint64_t value,
    const VkPipelineStageFlags stage_mask) {
    if (semaphore == VK_NULL_HANDLE || value == 0) {
        return;
    }
    pending_timeline_waits_.push_back(PendingTimelineWait{
        .semaphore = semaphore,
        .value = value,
        .stage_mask = stage_mask,
    });
}

void VulkanGSPipeline::endCommandBatch(bool use_fence,
                                       VkSemaphore signal_semaphore,
                                       std::uint64_t signal_value) {
    if (!commandBatchInProgress)
        _THROW_ERROR("No command batch in progress");
    CommandBatchSlot& slot = command_batch_slots_[active_command_batch_slot_];

    if (timestampNumWritten > 0) {
        [[maybe_unused]] auto cpu_timer = timeCpuStage("vksplat.command_batch.close_markers");
        while (timestampStackDepth > 0)
            PerfTimer::pushMarker(this);
    }

    {
        [[maybe_unused]] auto cpu_timer = timeCpuStage("vksplat.command_batch.vkEndCommandBuffer");
        if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
            _THROW_ERROR("Failed to end command buffer for batch");
        }
    }

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    std::vector<VkSemaphore> wait_semaphores;
    std::vector<VkPipelineStageFlags> wait_stages;
    std::vector<std::uint64_t> wait_values;
    std::vector<VkSemaphore> signal_semaphores;
    std::vector<std::uint64_t> signal_values;
    if (signal_semaphore != VK_NULL_HANDLE && signal_value != 0) {
        signal_semaphores.push_back(signal_semaphore);
        signal_values.push_back(signal_value);
    }

    VkTimelineSemaphoreSubmitInfo timeline_submit_info{};
    if (!pending_timeline_waits_.empty()) {
        wait_semaphores.reserve(pending_timeline_waits_.size());
        wait_stages.reserve(pending_timeline_waits_.size());
        wait_values.reserve(pending_timeline_waits_.size());
        for (const PendingTimelineWait& wait : pending_timeline_waits_) {
            wait_semaphores.push_back(wait.semaphore);
            wait_stages.push_back(wait.stage_mask);
            wait_values.push_back(wait.value);
        }
        timeline_submit_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_submit_info.waitSemaphoreValueCount = static_cast<uint32_t>(wait_values.size());
        timeline_submit_info.pWaitSemaphoreValues = wait_values.data();
        submit_info.waitSemaphoreCount = static_cast<uint32_t>(wait_semaphores.size());
        submit_info.pWaitSemaphores = wait_semaphores.data();
        submit_info.pWaitDstStageMask = wait_stages.data();
    }

    if (!signal_values.empty()) {
        timeline_submit_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_submit_info.signalSemaphoreValueCount = static_cast<uint32_t>(signal_values.size());
        timeline_submit_info.pSignalSemaphoreValues = signal_values.data();
        submit_info.signalSemaphoreCount = static_cast<uint32_t>(signal_semaphores.size());
        submit_info.pSignalSemaphores = signal_semaphores.data();
    }
    if (!wait_values.empty() || !signal_values.empty()) {
        submit_info.pNext = &timeline_submit_info;
    }

    {
        [[maybe_unused]] auto cpu_timer = timeCpuStage("vksplat.command_batch.vkQueueSubmit");
        if (vkQueueSubmit(command_queue, 1, &submit_info,
                          use_fence ? fence : VK_NULL_HANDLE) != VK_SUCCESS) {
            _THROW_ERROR("Failed to submit batch");
        }
    }
    pending_timeline_waits_.clear();

    commandBatchInProgress = false;

    if (use_fence) {
        [[maybe_unused]] auto cpu_timer = timeCpuStage("vksplat.command_batch.wait_fence");
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
    } else if (signal_semaphore == VK_NULL_HANDLE || signal_value == 0) {
        [[maybe_unused]] auto cpu_timer = timeCpuStage("vksplat.command_batch.wait_idle");
        vkQueueWaitIdle(command_queue);
    }

    PerfTimer::hostTic();

    if (!use_fence && signal_semaphore != VK_NULL_HANDLE && signal_value != 0) {
        slot.pending_signal = signal_semaphore;
        slot.pending_signal_value = signal_value;
        slot.pending_timestamp_count = timestampNumWritten;
        slot.pending_timestamp_marks = PerfTimer::takeMarkers();
        timestampNumWritten = 0;
        return;
    }

    if (timestampNumWritten > 0) {
        slot.pending_timestamp_marks = PerfTimer::takeMarkers();
        collectTimestampResults(slot, timestampNumWritten);
        slot.pending_timestamp_marks.clear();
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

void VulkanGSPipeline::addTimerCallback(TimerCallback callback) {
    timerCallbacks.push_back(std::move(callback));
}

void VulkanGSPipeline::setCpuTimerCallback(CpuTimerCallback callback) {
    cpuTimerCallback_ = std::move(callback);
}

VkAccessFlags2 toAccessMask(VulkanGSPipeline::BarrierMask barrierMask) {
    VkAccessFlags2 result = VK_ACCESS_2_NONE;
    if (barrierMask == VulkanGSPipeline::TRANSFER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_ACCESS_2_TRANSFER_READ_BIT;
    if (barrierMask == VulkanGSPipeline::TRANSFER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_ACCESS_2_SHADER_READ_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_ACCESS_2_SHADER_WRITE_BIT;
    if (barrierMask == VulkanGSPipeline::HOST_READ ||
        barrierMask == VulkanGSPipeline::HOST_READ_WRITE)
        result |= VK_ACCESS_2_HOST_READ_BIT;
    if (barrierMask == VulkanGSPipeline::HOST_WRITE ||
        barrierMask == VulkanGSPipeline::HOST_READ_WRITE)
        result |= VK_ACCESS_2_HOST_WRITE_BIT;
    if (barrierMask == VulkanGSPipeline::INDIRECT_DISPATCH_READ)
        result |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    return result;
}

VkPipelineStageFlags2 toStageMask(VulkanGSPipeline::BarrierMask barrierMask) {
    VkPipelineStageFlags2 result = VK_PIPELINE_STAGE_2_NONE;
    if (barrierMask == VulkanGSPipeline::TRANSFER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    if (barrierMask == VulkanGSPipeline::HOST_READ ||
        barrierMask == VulkanGSPipeline::HOST_WRITE ||
        barrierMask == VulkanGSPipeline::HOST_READ_WRITE)
        result |= VK_PIPELINE_STAGE_2_HOST_BIT;
    if (barrierMask == VulkanGSPipeline::INDIRECT_DISPATCH_READ)
        result |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    return result;
}

void VulkanGSPipeline::bufferMemoryBarrier(
    const std::vector<std::pair<_VulkanBuffer, VulkanGSPipeline::BarrierMask>>& buffers,
    VulkanGSPipeline::BarrierMask dstMask) {
    if (!commandBatchInProgress)
        return;

    const VkPipelineStageFlags2 dstStageMask = toStageMask(dstMask);
    const VkAccessFlags2 dstAccessMask = toAccessMask(dstMask);

    std::vector<VkBufferMemoryBarrier2> barriers;
    barriers.reserve(buffers.size());
    for (auto& [buffer, srcMask] : buffers) {
        if (buffer.buffer == VK_NULL_HANDLE)
            continue;
        VkBufferMemoryBarrier2 barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barrier.pNext = nullptr;
        barrier.srcStageMask = toStageMask(srcMask);
        barrier.srcAccessMask = toAccessMask(srcMask);
        barrier.dstStageMask = dstStageMask;
        barrier.dstAccessMask = dstAccessMask;
        barrier.srcQueueFamilyIndex = queue_family_index;
        barrier.dstQueueFamilyIndex = queue_family_index;
        barrier.buffer = buffer.buffer;
        barrier.offset = buffer.offset;
        barrier.size = buffer.size;
        barriers.push_back(barrier);
    }
    if (barriers.empty())
        return;

    const uint32_t barrier_count = static_cast<uint32_t>(barriers.size());
    VkDependencyInfo dependency = {};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.bufferMemoryBarrierCount = barrier_count;
    dependency.pBufferMemoryBarriers = barriers.data();

    vkCmdPipelineBarrier2(command_buffer, &dependency);
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

    const auto spirv_code = loadSpirv(spirv_path);
    recordSlangShaderBytecode(spirv_path, spirv_code.size() * sizeof(uint32_t));
    createShaderModule(spirv_code, &pipeline.shader);
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

    if (vkCreateComputePipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &pipeline.pipeline) != VK_SUCCESS)
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
            _CHECK_FATAL("Buffer " + std::to_string(binding) + " is NULL");
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
        _CHECK_FATAL("Indirect dispatch buffer is NULL");

    DEVICE_GUARD;

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

    const std::size_t num_buffers = pipeline.buffer_layouts.size();
    std::vector<VkDescriptorBufferInfo> buffer_infos(num_buffers);
    std::vector<VkWriteDescriptorSet> writes(num_buffers);
    for (std::size_t idx = 0; idx < num_buffers; ++idx) {
        const int binding = pipeline.buffer_layouts[idx];
        if (buffers[binding].buffer == VK_NULL_HANDLE)
            _CHECK_FATAL("Buffer " + std::to_string(binding) + " is NULL");
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

    vkCmdDispatchIndirect(command_buffer, indirect_buffer.buffer, indirect_buffer.offset + indirect_offset);
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
