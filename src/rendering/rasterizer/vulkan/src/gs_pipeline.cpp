#include "gs_pipeline.h"
#include "perf_timer.h"

#include "core/error.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
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

[[noreturn]] static void throwRendererContractViolation(std::string detail,
                                                        lfs::core::SourceSite site) {
    // Phase 7C-P4: shared typed helper (vulkan_result.hpp).
    lfs::rendering::throw_renderer_contract(std::move(detail), site);
}

[[nodiscard]] static const char* waitOutcomeLabel(
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

// Phase 7C-P3: throw dialect for gs_pipeline bounded waits (C1/C2). DeviceLost
// and other Result errors preserve their typed Error; flow outcomes map to
// Cancelled / Unavailable. Throws typed lfs::Exception (never stdout / untyped
// runtime_error).
[[noreturn]] static void throwBoundedWaitFailure(
    std::string detail,
    const lfs::Result<lfs::rendering::WaitOutcome>& outcome,
    const lfs::core::SourceSite site) {
    if (!outcome.has_value()) {
        throw lfs::Exception(outcome.error());
    }
    lfs::ErrorCode code = lfs::ErrorCode::Unavailable;
    switch (*outcome) {
    case lfs::rendering::WaitOutcome::Ready:
        code = lfs::ErrorCode::Internal;
        break;
    case lfs::rendering::WaitOutcome::Cancelled:
    case lfs::rendering::WaitOutcome::Shutdown:
        code = lfs::ErrorCode::Cancelled;
        break;
    case lfs::rendering::WaitOutcome::Quarantined:
        code = lfs::ErrorCode::Unavailable;
        break;
    }
    throw lfs::Exception(lfs::make_error(lfs::ErrorInit{
        .code = code,
        .domain = lfs::ErrorDomain::Vulkan,
        .user_message = "Vulkan GPU wait did not complete.",
        .detail = std::move(detail),
        .detection = site,
    }));
}

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
    if (!file) {
        lfs::rendering::throw_vk_setup(
            lfs::ErrorCode::NotFound,
            std::format(
            "VkSplat SPIR-V input could not be opened (path='{}', stream_good={}, stream_fail={})",
            spirv_path,
            file.good(),
            file.fail()),
            LFS_SOURCE_SITE_CURRENT());
    }

    std::streamsize fileSize = file.tellg();
    if (fileSize <= 0 || fileSize % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) {
        lfs::rendering::throw_vk_setup(
            lfs::ErrorCode::InvalidArgument,
            std::format(
            "VkSplat SPIR-V input must be non-empty and uint32-aligned (path='{}', observed_bytes={}, word_bytes={}, remainder={})",
            spirv_path,
            fileSize,
            sizeof(uint32_t),
            fileSize > 0 ? fileSize % static_cast<std::streamsize>(sizeof(uint32_t)) : fileSize),
            LFS_SOURCE_SITE_CURRENT());
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> spirv_code(fileSize / sizeof(uint32_t));
    if (!file.read(reinterpret_cast<char*>(spirv_code.data()), fileSize)) {
        lfs::rendering::throw_vk_setup(
            lfs::ErrorCode::Internal,
            std::format(
            "VkSplat SPIR-V input read was incomplete (path='{}', requested_bytes={}, observed_bytes={}, stream_fail={}, stream_bad={})",
            spirv_path,
            fileSize,
            file.gcount(),
            file.fail(),
            file.bad()),
            LFS_SOURCE_SITE_CURRENT());
    }

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
    vulkan_dispatch_ = lfs::rendering::VulkanDispatch::real();
}

void VulkanGSPipeline::setVulkanDispatch(lfs::rendering::VulkanDispatch dispatch) noexcept {
    vulkan_dispatch_ = std::move(dispatch);
}

const lfs::rendering::VulkanDispatch& VulkanGSPipeline::vulkanDispatch() const noexcept {
    return vulkan_dispatch_;
}

const lfs::rendering::SubmissionState& VulkanGSPipeline::lastSubmissionState() const noexcept {
    return last_submission_state_;
}

VulkanGSPipeline::~VulkanGSPipeline() noexcept {
    cancelCommandBatch();
    try {
        cleanup();
    } catch (const std::exception& error) {
        fprintf(stderr, "VulkanGSPipeline cleanup failed: %s\n", error.what());
    } catch (...) {
        fprintf(stderr, "VulkanGSPipeline cleanup failed with an unknown error\n");
    }
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
        lfs::rendering::throw_renderer_contract(
            std::format(
            "VkSplat external initialization requires valid Vulkan objects (instance={:#x}, physical_device={:#x}, device={:#x}, queue={:#x}, queue_family={}, allocator={:#x}, pipeline_cache={:#x})",
            lfs::rendering::vkHandleValue(external_instance),
            lfs::rendering::vkHandleValue(external_physical_device),
            lfs::rendering::vkHandleValue(external_device),
            lfs::rendering::vkHandleValue(external_queue),
            external_queue_family_index,
            lfs::rendering::vkHandleValue(external_allocator),
            lfs::rendering::vkHandleValue(external_pipeline_cache)),
            LFS_SOURCE_SITE_CURRENT());
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
        lfs::rendering::throw_renderer_contract(
            std::format(
            "VkSplat requires vkCmdPushDescriptorSetKHR, but the device proc address is null (device={:#x}, queue_family={})",
            lfs::rendering::vkHandleValue(device),
            queue_family_index),
            LFS_SOURCE_SITE_CURRENT());
    }
    debug_name_writer_.initialize(device);

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
    _(page_frames)
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
    _(survivors)
    _(survivor_state)
    _(visible_emit_count)
    _(orig_ids)
    _(cumsum_counts)
    _(visible_dispatch)
    _(macro_partials)
    _(macro_active_mask)
    _(macro_wave_args)
    _(index_buffer_offset)
    _(sorting_keys_1)
    _(sorting_keys_2)
    _(sorting_gauss_idx_1)
    _(sorting_gauss_idx_2)
    _(tile_sort_count)
    _(tile_sort_dispatch_args)
    _(tile_ranges)
    _(tile_batch_counts)
    _(tile_batch_offsets)
    _(tile_batch_dispatch_args)
    _(tile_batch_descriptors)
    _(tile_batch_pixel_state)
    _(tile_batch_n_contributors)
    _(pixel_state)
    _(pixel_depth)
    _(n_contributors)
    _(_cumsum_blockSums)
    _(_cumsum_blockSums2)
    _(_sorting_histogram)
    _(_sorting_histogram_cumsum)
    _(lod_indices)
    _(lod_logical_indices)
    _(lod_levels)
    _(lod_weights)
    _(lod_gpu_indices)
    _(lod_gpu_logical_indices)
    _(lod_gpu_weights)
    _(lod_gpu_counts)
    _(lod_chunk_touch)
    _(lod_compact_counts)
    _(lod_compact_protected)
    _(lod_compact_misses)
    _(lod_gpu_levels)
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
    _(page_frames)
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
    _(survivors)
    _(survivor_state)
    _(visible_emit_count)
    _(orig_ids)
    _(cumsum_counts)
    _(visible_dispatch)
    _(macro_partials)
    _(macro_active_mask)
    _(macro_wave_args)
    _(index_buffer_offset)
    _(sorting_keys_1)
    _(sorting_keys_2)
    _(sorting_gauss_idx_1)
    _(sorting_gauss_idx_2)
    _(tile_sort_count)
    _(tile_sort_dispatch_args)
    _(tile_ranges)
    _(tile_batch_counts)
    _(tile_batch_offsets)
    _(tile_batch_dispatch_args)
    _(tile_batch_descriptors)
    _(tile_batch_pixel_state)
    _(tile_batch_n_contributors)
    _(pixel_state)
    _(pixel_depth)
    _(n_contributors)
    _(_cumsum_blockSums)
    _(_cumsum_blockSums2)
    _(_sorting_histogram)
    _(_sorting_histogram_cumsum)
    _(lod_indices)
    _(lod_logical_indices)
    _(lod_levels)
    _(lod_weights)
    _(lod_gpu_indices)
    _(lod_gpu_logical_indices)
    _(lod_gpu_weights)
    _(lod_gpu_counts)
    _(lod_chunk_touch)
    _(lod_compact_counts)
    _(lod_compact_protected)
    _(lod_compact_misses)
    _(lod_gpu_levels)
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
        const VkResult idle_result = vkDeviceWaitIdle(device);
        if (idle_result != VK_SUCCESS) {
            lfs::rendering::throw_vk_result(
                idle_result,
                "vkDeviceWaitIdle",
                std::format(
                "VkSplat cleanup could not retire device work before destroying resources (device={:#x}, result={}({}))",
                lfs::rendering::vkHandleValue(device),
                lfs::rendering::vkResultToString(idle_result),
                static_cast<int>(idle_result)),
                LFS_SOURCE_SITE_CURRENT());
        }

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
    last_timeline_wait_values_.clear();
    last_timeline_signal_values_.clear();
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
    vk_cmd_push_descriptor_set_ = nullptr;
    debug_name_writer_.reset();
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

void VulkanGSPipeline::setDebugObjectName(const VkObjectType type,
                                          const std::uint64_t handle,
                                          const std::string_view name) const {
    const VkResult result = debug_name_writer_.set(type, handle, name);
    if (result != VK_SUCCESS) {
        fprintf(stderr,
                "vkSetDebugUtilsObjectNameEXT failed for '%.*s' (type=%d, handle=0x%llx, result=%s(%d)) at %s:%d\n",
                static_cast<int>(name.size()),
                name.data(),
                static_cast<int>(type),
                static_cast<unsigned long long>(handle),
                lfs::rendering::vkResultToString(result),
                static_cast<int>(result),
                __FILE__,
                __LINE__);
    }
}

void VulkanGSPipeline::validateBufferRange(const _VulkanBuffer& buffer,
                                           const VkDeviceSize relative_offset,
                                           const VkDeviceSize size,
                                           const std::string_view operation) const {
    if (!buffer.containsRange(relative_offset, size)) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "{} requires a non-null buffer and a byte range within both its view and backing buffer (buffer={:#x}, allocation={:#x}, base_offset={}, relative_offset={}, size={}, view_capacity={}, backing_size={}, label='{}')",
            operation,
            lfs::rendering::vkHandleValue(buffer.buffer),
            lfs::rendering::vkHandleValue(buffer.allocation),
            buffer.offset,
            relative_offset,
            size,
            buffer.capacity,
            buffer.allocSize,
            buffer.label ? buffer.label : "<unlabeled>"),
            LFS_SOURCE_SITE_CURRENT());
    }
}

void VulkanGSPipeline::validateFillRange(const _VulkanBuffer& buffer,
                                         const VkDeviceSize relative_offset,
                                         const VkDeviceSize size,
                                         const std::string_view operation) const {
    validateBufferRange(buffer, relative_offset, size, operation);
    const VkDeviceSize absolute_offset = buffer.offset + relative_offset;
    if ((absolute_offset & 3u) != 0 || (size & 3u) != 0) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "{} requires four-byte-aligned vkCmdFillBuffer bounds (buffer={:#x}, absolute_offset={}, size={}, offset_mod4={}, size_mod4={}, allocation_size={}, label='{}')",
            operation,
            lfs::rendering::vkHandleValue(buffer.buffer),
            absolute_offset,
            size,
            absolute_offset & 3u,
            size & 3u,
            buffer.allocSize,
            buffer.label ? buffer.label : "<unlabeled>"),
            LFS_SOURCE_SITE_CURRENT());
    }
}

void VulkanGSPipeline::createCommandPool() {
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family_index;

    const VkResult pool_result = vkCreateCommandPool(device, &pool_info, nullptr, &command_pool);
    if (pool_result != VK_SUCCESS) {
        lfs::rendering::throw_vk_result(
            pool_result,
            "vkCreateCommandPool",
            std::format(
            "VkSplat command-pool creation failed (device={:#x}, queue_family={}, flags={:#x}, result={}({}))",
            lfs::rendering::vkHandleValue(device),
            queue_family_index,
            static_cast<std::uint32_t>(pool_info.flags),
            lfs::rendering::vkResultToString(pool_result),
            static_cast<int>(pool_result)),
            LFS_SOURCE_SITE_CURRENT());
    }
    setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL, command_pool, "vksplat.command_pool");

    std::array<VkCommandBuffer, kCommandBatchSlotCount> command_buffers{};
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool;
    alloc_info.commandBufferCount = kCommandBatchSlotCount;

    const VkResult command_result = vkAllocateCommandBuffers(device, &alloc_info, command_buffers.data());
    if (command_result != VK_SUCCESS) {
        lfs::rendering::throw_vk_result(
            command_result,
            "vkAllocateCommandBuffers",
            std::format(
            "VkSplat command-buffer allocation failed (device={:#x}, command_pool={:#x}, requested_count={}, result={}({}))",
            lfs::rendering::vkHandleValue(device),
            lfs::rendering::vkHandleValue(command_pool),
            kCommandBatchSlotCount,
            lfs::rendering::vkResultToString(command_result),
            static_cast<int>(command_result)),
            LFS_SOURCE_SITE_CURRENT());
    }
    for (std::uint32_t i = 0; i < kCommandBatchSlotCount; ++i) {
        command_batch_slots_[i].command_buffer = command_buffers[i];
        if (debug_name_writer_.enabled()) {
            setDebugObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER,
                               command_buffers[i],
                               std::format("vksplat.command[{}]", i));
        }
    }
    command_buffer = command_batch_slots_[0].command_buffer;
}

void VulkanGSPipeline::createFence() {
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;

    VkResult result = vkCreateFence(device, &fenceInfo, nullptr, &fence);
    if (result != VK_SUCCESS) {
        lfs::rendering::throw_vk_result(
            result,
            "vkCreateFence",
            std::format(
            "VkSplat batch fence creation failed (device={:#x}, flags={:#x}, result={}({}))",
            lfs::rendering::vkHandleValue(device),
            static_cast<std::uint32_t>(fenceInfo.flags),
            lfs::rendering::vkResultToString(result),
            static_cast<int>(result)),
            LFS_SOURCE_SITE_CURRENT());
    }
    setDebugObjectName(VK_OBJECT_TYPE_FENCE, fence, "vksplat.batch.fence");
}

void VulkanGSPipeline::createQueryPools() {
    // timestamp
    VkQueryPoolCreateInfo queryPoolCreateInfo = {};
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = MAX_TIMESTAMP_QUERY_COUNT;
    for (std::size_t i = 0; i < command_batch_slots_.size(); ++i) {
        CommandBatchSlot& slot = command_batch_slots_[i];
        const VkResult result = vkCreateQueryPool(device, &queryPoolCreateInfo, nullptr, &slot.timestamp_query_pool);
        if (result != VK_SUCCESS) {
            lfs::rendering::throw_vk_result(
                result,
                "vkCreateQueryPool",
                std::format(
                "VkSplat timestamp query-pool creation failed (slot={}, requested_queries={}, device={:#x}, result={}({}))",
                i,
                MAX_TIMESTAMP_QUERY_COUNT,
                lfs::rendering::vkHandleValue(device),
                lfs::rendering::vkResultToString(result),
                static_cast<int>(result)),
                LFS_SOURCE_SITE_CURRENT());
        }
        if (debug_name_writer_.enabled()) {
            setDebugObjectName(VK_OBJECT_TYPE_QUERY_POOL,
                               slot.timestamp_query_pool,
                               std::format("vksplat.timestamp_queries[{}]", i));
        }
    }
    timestamp_query_pool = command_batch_slots_[0].timestamp_query_pool;
}

void VulkanGSPipeline::createShaderModule(const std::vector<uint32_t>& spirv_code, VkShaderModule* pShaderModule) {
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_code.size() * sizeof(uint32_t);
    create_info.pCode = spirv_code.data();

    const VkResult result = vkCreateShaderModule(device, &create_info, nullptr, pShaderModule);
    if (result != VK_SUCCESS) {
        lfs::rendering::throw_vk_result(
            result,
            "vkCreateShaderModule",
            std::format(
            "VkSplat shader-module creation failed (device={:#x}, byte_size={}, output={:#x}, result={}({}))",
            lfs::rendering::vkHandleValue(device),
            create_info.codeSize,
            lfs::rendering::vkHandleValue(*pShaderModule),
            lfs::rendering::vkResultToString(result),
            static_cast<int>(result)),
            LFS_SOURCE_SITE_CURRENT());
    }
}

void VulkanGSPipeline::beginCommandBatch() {
    if (commandBatchInProgress) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "beginCommandBatch cannot begin twice (batch_active={}, active_slot={}, next_slot={}, command_buffer={:#x})",
            commandBatchInProgress,
            active_command_batch_slot_,
            next_command_batch_slot_,
            lfs::rendering::vkHandleValue(command_buffer)),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (device == VK_NULL_HANDLE || command_queue == VK_NULL_HANDLE || command_pool == VK_NULL_HANDLE) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "beginCommandBatch requires initialized Vulkan queue resources (device={:#x}, queue={:#x}, command_pool={:#x}, next_slot={})",
            lfs::rendering::vkHandleValue(device),
            lfs::rendering::vkHandleValue(command_queue),
            lfs::rendering::vkHandleValue(command_pool),
            next_command_batch_slot_),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (next_command_batch_slot_ >= command_batch_slots_.size()) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "beginCommandBatch next slot is outside the command ring (next_slot={}, ring_size={})",
            next_command_batch_slot_,
            command_batch_slots_.size()),
            LFS_SOURCE_SITE_CURRENT());
    }

    active_command_batch_slot_ = next_command_batch_slot_;
    next_command_batch_slot_ = (next_command_batch_slot_ + 1) % kCommandBatchSlotCount;
    CommandBatchSlot& slot = command_batch_slots_[active_command_batch_slot_];
    waitForPendingBatchSlot(slot);
    command_buffer = slot.command_buffer;
    timestamp_query_pool = slot.timestamp_query_pool;
    if (command_buffer == VK_NULL_HANDLE || timestamp_query_pool == VK_NULL_HANDLE) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "beginCommandBatch selected an incomplete command slot (slot={}, ring_size={}, command_buffer={:#x}, timestamp_query_pool={:#x})",
            active_command_batch_slot_,
            command_batch_slots_.size(),
            lfs::rendering::vkHandleValue(command_buffer),
            lfs::rendering::vkHandleValue(timestamp_query_pool)),
            LFS_SOURCE_SITE_CURRENT());
    }
    timestampNumWritten = 0;
    timestampStackDepth = 0;
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vulkan_dispatch_.begin_command_buffer == nullptr ||
        vulkan_dispatch_.cmd_pipeline_barrier2 == nullptr ||
        vulkan_dispatch_.cmd_reset_query_pool == nullptr) {
        throwRendererContractViolation(
            "beginCommandBatch requires a complete VulkanDispatch begin prologue "
            "(begin_command_buffer, cmd_pipeline_barrier2, cmd_reset_query_pool)",
            LFS_SOURCE_SITE_CURRENT());
    }

    const VkResult begin_result =
        vulkan_dispatch_.begin_command_buffer(command_buffer, &begin_info);
    if (begin_result != VK_SUCCESS) {
        lfs::rendering::throw_vk_result(
            begin_result,
            "vkBeginCommandBuffer",
            std::format(
            "vkBeginCommandBuffer could not start the VkSplat batch (slot={}, command_buffer={:#x}, flags={:#x}, result={}({}))",
            active_command_batch_slot_,
            lfs::rendering::vkHandleValue(command_buffer),
            static_cast<std::uint32_t>(begin_info.flags),
            lfs::rendering::vkResultToString(begin_result),
            static_cast<int>(begin_result)),
            LFS_SOURCE_SITE_CURRENT());
    }

    // Command buffers rotate, but the VkSplat scratch, indirect, output, and
    // readback storage is shared across slots. Queue submission order alone
    // does not prevent adjacent batches from overlapping those accesses. A
    // queue-local dependency at the batch boundary covers precisely the stages
    // used by this compute rasterizer; CUDA/Vulkan ownership edges remain on
    // their dedicated external semaphores.
    constexpr VkPipelineStageFlags2 kRasterizerStages =
        VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT |
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    constexpr VkAccessFlags2 kRasterizerAccesses =
        VK_ACCESS_2_TRANSFER_READ_BIT |
        VK_ACCESS_2_TRANSFER_WRITE_BIT |
        VK_ACCESS_2_SHADER_READ_BIT |
        VK_ACCESS_2_SHADER_WRITE_BIT |
        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    const VkMemoryBarrier2 reuse_barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = kRasterizerStages,
        .srcAccessMask = kRasterizerAccesses,
        .dstStageMask = kRasterizerStages,
        .dstAccessMask = kRasterizerAccesses,
    };
    const VkDependencyInfo reuse_dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = 0,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &reuse_barrier,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = 0,
        .pImageMemoryBarriers = nullptr,
    };
    vulkan_dispatch_.cmd_pipeline_barrier2(command_buffer, &reuse_dependency);

    commandBatchInProgress = true;
    try {
        PerfTimer::hostToc();
        vulkan_dispatch_.cmd_reset_query_pool(
            command_buffer, timestamp_query_pool, 0, MAX_TIMESTAMP_QUERY_COUNT);
        PerfTimer::popMarkers(this);
    } catch (...) {
        cancelCommandBatch();
        throw;
    }
}

void VulkanGSPipeline::cancelCommandBatch() noexcept {
    const bool was_recording = commandBatchInProgress;
    commandBatchInProgress = false;
    pending_timeline_waits_.clear();
    timestampNumWritten = 0;
    timestampStackDepth = 0;
    PerfTimer::discardMarkers();

    if (!was_recording)
        return;

    if (device != VK_NULL_HANDLE && command_buffer != VK_NULL_HANDLE &&
        vulkan_dispatch_.reset_command_buffer != nullptr) {
        const VkResult result = vulkan_dispatch_.reset_command_buffer(command_buffer, 0);
        if (result != VK_SUCCESS) {
            fprintf(stderr,
                    "cancelCommandBatch could not reset the cancelled command buffer (slot=%u, command_buffer=0x%llx, result=%s(%d)) at %s:%d\n",
                    active_command_batch_slot_,
                    static_cast<unsigned long long>(lfs::rendering::vkHandleValue(command_buffer)),
                    lfs::rendering::vkResultToString(result),
                    static_cast<int>(result),
                    __FILE__,
                    __LINE__);
        }
    }
    try {
        PerfTimer::hostTic();
    } catch (const std::exception& error) {
        fprintf(stderr, "Failed to restore Vulkan host timer after batch cancellation: %s\n", error.what());
    } catch (...) {
        fprintf(stderr, "Failed to restore Vulkan host timer after batch cancellation\n");
    }
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
    if (vulkan_dispatch_.get_semaphore_counter_value == nullptr) {
        throwRendererContractViolation(
            std::format(
                "vkGetSemaphoreCounterValue dispatch is null while polling a VkSplat batch (semaphore={:#x}, requested_value={})",
                lfs::rendering::vkHandleValue(semaphore),
                value),
            LFS_SOURCE_SITE_CURRENT());
    }
    std::uint64_t completed_value = 0;
    const VkResult result =
        vulkan_dispatch_.get_semaphore_counter_value(device, semaphore, &completed_value);
    if (result != VK_SUCCESS) {
        lfs::rendering::throw_vk_result(
            result,
            "vkGetSemaphoreCounterValue",
            std::format(
            "vkGetSemaphoreCounterValue failed while polling a VkSplat batch (semaphore={:#x}, requested_value={}, completed_value={}, result={}({}))",
            lfs::rendering::vkHandleValue(semaphore),
            value,
            completed_value,
            lfs::rendering::vkResultToString(result),
            static_cast<int>(result)),
            LFS_SOURCE_SITE_CURRENT());
    }
    return completed_value >= value;
}

bool VulkanGSPipeline::wasTimelineSignalSubmitted(const VkSemaphore semaphore,
                                                  const std::uint64_t value) const noexcept {
    if (semaphore == VK_NULL_HANDLE || value == 0) {
        return false;
    }
    const auto it = last_timeline_signal_values_.find(semaphore);
    return it != last_timeline_signal_values_.end() && it->second >= value;
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
            // Phase 7C-P3 C1: bounded wait via dispatch. Non-Ready throws and
            // leaves pending_signal set (no manufactured readiness).
            lfs::rendering::WaitContext wait_ctx;
            wait_ctx.dispatch = &vulkan_dispatch_;
            wait_ctx.fingerprint = "vksplat.pipeline.wait_pending_slot";
            wait_ctx.owner_quarantine_flag = &gpu_wait_quarantined_;
            auto wait_outcome = lfs::rendering::wait_semaphores_bounded(
                device,
                wait_info,
                std::stop_token{},
                lfs::rendering::VulkanWaitPolicy{},
                wait_ctx);
            if (!wait_outcome.has_value() ||
                *wait_outcome != lfs::rendering::WaitOutcome::Ready) {
                const std::string outcome_label = wait_outcome.has_value()
                                                      ? std::string(waitOutcomeLabel(*wait_outcome))
                                                      : std::string(wait_outcome.error().detail());
                throwBoundedWaitFailure(
                    std::format(
                        "wait_semaphores_bounded failed while retiring a VkSplat batch slot "
                        "(semaphore={:#x}, value={}, slot_command_buffer={:#x}, outcome={})",
                        lfs::rendering::vkHandleValue(slot.pending_signal),
                        slot.pending_signal_value,
                        lfs::rendering::vkHandleValue(slot.command_buffer),
                        outcome_label),
                    wait_outcome,
                    LFS_SOURCE_SITE_CURRENT());
            }
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
    const VkResult result = vkGetQueryPoolResults(
        device, slot.timestamp_query_pool,
        0, timestamp_count,
        sizeof(uint64_t) * timestamp_count,
        timestamps.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result != VK_SUCCESS) {
        lfs::rendering::throw_vk_result(
            result,
            "vkGetQueryPoolResults",
            std::format(
            "vkGetQueryPoolResults failed for VkSplat timestamps (query_pool={:#x}, requested_count={}, max_count={}, result={}({}))",
            lfs::rendering::vkHandleValue(slot.timestamp_query_pool),
            timestamp_count,
            MAX_TIMESTAMP_QUERY_COUNT,
            lfs::rendering::vkResultToString(result),
            static_cast<int>(result)),
            LFS_SOURCE_SITE_CURRENT());
    }
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
    if (semaphore == VK_NULL_HANDLE || value == 0 || stage_mask == 0) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "addTimelineWait requires a valid timeline edge (semaphore={:#x}, value={}, stage_mask={:#x}, pending_waits={})",
            lfs::rendering::vkHandleValue(semaphore),
            value,
            static_cast<std::uint64_t>(stage_mask),
            pending_timeline_waits_.size()),
            LFS_SOURCE_SITE_CURRENT());
    }
    const std::uint64_t previous = last_timeline_wait_values_[semaphore];
    if (value <= previous) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "VkSplat Vulkan timeline waits must increase strictly (semaphore={:#x}, requested_value={}, previous_value={}, pending_waits={})",
            lfs::rendering::vkHandleValue(semaphore),
            value,
            previous,
            pending_timeline_waits_.size()),
            LFS_SOURCE_SITE_CURRENT());
    }
    last_timeline_wait_values_[semaphore] = value;
    pending_timeline_waits_.push_back(PendingTimelineWait{
        .semaphore = semaphore,
        .value = value,
        .stage_mask = stage_mask,
    });
}

void VulkanGSPipeline::endCommandBatch(bool use_fence,
                                       VkSemaphore signal_semaphore,
                                       std::uint64_t signal_value,
                                       VkSemaphore secondary_signal_semaphore,
                                       std::uint64_t secondary_signal_value) {
    if (!commandBatchInProgress) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "endCommandBatch called with no active batch (batch_active={}, active_slot={}, next_slot={}, command_buffer={:#x})",
            commandBatchInProgress,
            active_command_batch_slot_,
            next_command_batch_slot_,
            lfs::rendering::vkHandleValue(command_buffer)),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (active_command_batch_slot_ >= command_batch_slots_.size()) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "endCommandBatch active slot is outside the command ring (active_slot={}, ring_size={}, command_buffer={:#x})",
            active_command_batch_slot_,
            command_batch_slots_.size(),
            lfs::rendering::vkHandleValue(command_buffer)),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (command_buffer == VK_NULL_HANDLE || command_queue == VK_NULL_HANDLE) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "endCommandBatch requires a recorded command buffer and submission queue (command_buffer={:#x}, queue={:#x}, active_slot={})",
            lfs::rendering::vkHandleValue(command_buffer),
            lfs::rendering::vkHandleValue(command_queue),
            active_command_batch_slot_),
            LFS_SOURCE_SITE_CURRENT());
    }
    if ((signal_semaphore == VK_NULL_HANDLE) != (signal_value == 0)) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "endCommandBatch timeline signal handle/value must be supplied together (semaphore={:#x}, value={}, use_fence={}, active_slot={})",
            lfs::rendering::vkHandleValue(signal_semaphore),
            signal_value,
            use_fence,
            active_command_batch_slot_),
            LFS_SOURCE_SITE_CURRENT());
    }
    if ((secondary_signal_semaphore == VK_NULL_HANDLE) != (secondary_signal_value == 0)) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "endCommandBatch secondary timeline signal handle/value must be supplied together (semaphore={:#x}, value={}, use_fence={}, active_slot={})",
            lfs::rendering::vkHandleValue(secondary_signal_semaphore),
            secondary_signal_value,
            use_fence,
            active_command_batch_slot_),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (signal_semaphore != VK_NULL_HANDLE &&
        signal_semaphore == secondary_signal_semaphore) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "endCommandBatch timeline signal handles must be distinct (primary={:#x}, secondary={:#x}, primary_value={}, secondary_value={}, active_slot={})",
            lfs::rendering::vkHandleValue(signal_semaphore),
            lfs::rendering::vkHandleValue(secondary_signal_semaphore),
            signal_value,
            secondary_signal_value,
            active_command_batch_slot_),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (use_fence && fence == VK_NULL_HANDLE) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "endCommandBatch requested fence completion with a null fence (use_fence={}, fence={:#x}, active_slot={})",
            use_fence,
            lfs::rendering::vkHandleValue(fence),
            active_command_batch_slot_),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (signal_semaphore != VK_NULL_HANDLE) {
        const std::uint64_t previous = last_timeline_signal_values_[signal_semaphore];
        if (signal_value <= previous) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                "VkSplat Vulkan timeline signals must increase strictly (semaphore={:#x}, signal_value={}, previous_value={}, active_slot={})",
                lfs::rendering::vkHandleValue(signal_semaphore),
                signal_value,
                previous,
                active_command_batch_slot_),
                LFS_SOURCE_SITE_CURRENT());
        }
    }
    if (secondary_signal_semaphore != VK_NULL_HANDLE) {
        const std::uint64_t previous =
            last_timeline_signal_values_[secondary_signal_semaphore];
        if (secondary_signal_value <= previous) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                "VkSplat secondary Vulkan timeline signals must increase strictly (semaphore={:#x}, signal_value={}, previous_value={}, active_slot={})",
                lfs::rendering::vkHandleValue(secondary_signal_semaphore),
                secondary_signal_value,
                previous,
                active_command_batch_slot_),
                LFS_SOURCE_SITE_CURRENT());
        }
    }
    CommandBatchSlot& slot = command_batch_slots_[active_command_batch_slot_];

    // T0 — begin lifecycle for this submit (no-reset / no-replacement row).
    {
        using lfs::rendering::SubmissionFencePolicy;
        using lfs::rendering::SubmissionTransition;
        auto begin = lfs::rendering::apply_submission_transition(
            last_submission_state_,
            SubmissionTransition::BeginLifecycle,
            SubmissionFencePolicy::NoResetNoReplacement,
            signal_value);
        if (!begin) {
            cancelCommandBatch();
            throwRendererContractViolation(
                std::format(
                    "SubmissionState T0 BeginLifecycle failed (detail={}, signal_value={})",
                    begin.error().detail(),
                    signal_value),
                LFS_SOURCE_SITE_CURRENT());
        }
    }

    if (timestampNumWritten > 0) {
        [[maybe_unused]] auto cpu_timer = timeCpuStage("vksplat.command_batch.close_markers");
        while (timestampStackDepth > 0)
            PerfTimer::pushMarker(this);
    }

    if (vulkan_dispatch_.end_command_buffer == nullptr ||
        vulkan_dispatch_.queue_submit == nullptr) {
        cancelCommandBatch();
        throwRendererContractViolation(
            "endCommandBatch requires VulkanDispatch end_command_buffer and queue_submit",
            LFS_SOURCE_SITE_CURRENT());
    }

    {
        [[maybe_unused]] auto cpu_timer = timeCpuStage("vksplat.command_batch.vkEndCommandBuffer");
        const VkResult end_result = vulkan_dispatch_.end_command_buffer(command_buffer);
        if (end_result != VK_SUCCESS) {
            cancelCommandBatch();
            lfs::rendering::throw_vk_result(
                end_result,
                "vkEndCommandBuffer",
                std::format(
                "vkEndCommandBuffer failed for the active VkSplat batch (slot={}, command_buffer={:#x}, result={}({}))",
                active_command_batch_slot_,
                lfs::rendering::vkHandleValue(command_buffer),
                lfs::rendering::vkResultToString(end_result),
                static_cast<int>(end_result)),
                LFS_SOURCE_SITE_CURRENT());
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
    if (secondary_signal_semaphore != VK_NULL_HANDLE && secondary_signal_value != 0) {
        signal_semaphores.push_back(secondary_signal_semaphore);
        signal_values.push_back(secondary_signal_value);
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

    if (wait_semaphores.size() != wait_stages.size() ||
        wait_semaphores.size() != wait_values.size() ||
        signal_semaphores.size() != signal_values.size() ||
        timeline_submit_info.waitSemaphoreValueCount != submit_info.waitSemaphoreCount ||
        timeline_submit_info.signalSemaphoreValueCount != submit_info.signalSemaphoreCount) {
        cancelCommandBatch();
        lfs::rendering::throw_renderer_contract(
            std::format(
            "VkSplat submit semaphore arrays disagree with VkSubmitInfo counts (wait_handles={}, wait_stages={}, wait_values={}, submit_wait_count={}, timeline_wait_count={}, signal_handles={}, signal_values={}, submit_signal_count={}, timeline_signal_count={})",
            wait_semaphores.size(),
            wait_stages.size(),
            wait_values.size(),
            submit_info.waitSemaphoreCount,
            timeline_submit_info.waitSemaphoreValueCount,
            signal_semaphores.size(),
            signal_values.size(),
            submit_info.signalSemaphoreCount,
            timeline_submit_info.signalSemaphoreValueCount),
            LFS_SOURCE_SITE_CURRENT());
    }

    {
        [[maybe_unused]] auto cpu_timer = timeCpuStage("vksplat.command_batch.vkQueueSubmit");
        const VkResult submit_result = vulkan_dispatch_.queue_submit(
            command_queue, 1, &submit_info, use_fence ? fence : VK_NULL_HANDLE);
        if (submit_result != VK_SUCCESS) {
            // T3 — rejected submit: no publish, no replace (no-reset row).
            {
                using lfs::rendering::SubmissionFencePolicy;
                using lfs::rendering::SubmissionTransition;
                auto rejected = lfs::rendering::apply_submission_transition(
                    last_submission_state_,
                    SubmissionTransition::SubmitRejected,
                    SubmissionFencePolicy::NoResetNoReplacement);
                if (!rejected) {
                    // Still cancel + throw the original submit failure; surface
                    // the contract violation only in the primary message.
                    cancelCommandBatch();
                    throwRendererContractViolation(
                        std::format(
                            "vkQueueSubmit failed and SubmissionState T3 rejected "
                            "(submit_result={}({}), transition_detail={})",
                            lfs::rendering::vkResultToString(submit_result),
                            static_cast<int>(submit_result),
                            rejected.error().detail()),
                        LFS_SOURCE_SITE_CURRENT());
                }
            }
            cancelCommandBatch();
            lfs::rendering::throw_vk_result(
                submit_result,
                "vkQueueSubmit",
                std::format(
                "vkQueueSubmit failed for the VkSplat batch (queue={:#x}, command_buffer={:#x}, slot={}, wait_count={}, signal_count={}, fence={:#x}, result={}({}))",
                lfs::rendering::vkHandleValue(command_queue),
                lfs::rendering::vkHandleValue(command_buffer),
                active_command_batch_slot_,
                submit_info.waitSemaphoreCount,
                submit_info.signalSemaphoreCount,
                lfs::rendering::vkHandleValue(use_fence ? fence : VK_NULL_HANDLE),
                lfs::rendering::vkResultToString(submit_result),
                static_cast<int>(submit_result)),
                LFS_SOURCE_SITE_CURRENT());
        }
    }

    // T4 — set submit_accepted BEFORE publishing the candidate (ordering, not
    // atomicity — Amendment 1).
    {
        using lfs::rendering::SubmissionFencePolicy;
        using lfs::rendering::SubmissionTransition;
        auto accepted = lfs::rendering::apply_submission_transition(
            last_submission_state_,
            SubmissionTransition::SubmitAccepted,
            SubmissionFencePolicy::NoResetNoReplacement,
            signal_value);
        if (!accepted) {
            cancelCommandBatch();
            throwRendererContractViolation(
                std::format(
                    "SubmissionState T4 SubmitAccepted failed after native submit "
                    "success (detail={})",
                    accepted.error().detail()),
                LFS_SOURCE_SITE_CURRENT());
        }
    }

    // T5 — publish once (host-side evidence map). Never host-signal.
    if (signal_semaphore != VK_NULL_HANDLE) {
        last_timeline_signal_values_[signal_semaphore] = signal_value;
    }
    if (secondary_signal_semaphore != VK_NULL_HANDLE) {
        last_timeline_signal_values_[secondary_signal_semaphore] = secondary_signal_value;
    }
    if (signal_semaphore != VK_NULL_HANDLE && signal_value != 0) {
        using lfs::rendering::SubmissionFencePolicy;
        using lfs::rendering::SubmissionTransition;
        auto published = lfs::rendering::apply_submission_transition(
            last_submission_state_,
            SubmissionTransition::PublishTimeline,
            SubmissionFencePolicy::NoResetNoReplacement,
            signal_value);
        if (!published) {
            // Publication map already written; this is a bookkeeping fault.
            throwRendererContractViolation(
                std::format(
                    "SubmissionState T5 PublishTimeline failed after host publication "
                    "(detail={}, signal_value={})",
                    published.error().detail(),
                    signal_value),
                LFS_SOURCE_SITE_CURRENT());
        }
    }
    pending_timeline_waits_.clear();

    commandBatchInProgress = false;

    if (use_fence) {
        [[maybe_unused]] auto cpu_timer = timeCpuStage("vksplat.command_batch.wait_fence");
        if (vulkan_dispatch_.reset_fences == nullptr) {
            PerfTimer::discardMarkers();
            timestampNumWritten = 0;
            PerfTimer::hostTic();
            throwRendererContractViolation(
                "endCommandBatch fence path requires VulkanDispatch reset_fences",
                LFS_SOURCE_SITE_CURRENT());
        }
        // Phase 7C-P3 C2: bounded post-submit fence wait via dispatch.
        // Non-Ready throws without reset (reset implies free-for-reuse).
        lfs::rendering::WaitContext wait_ctx;
        wait_ctx.dispatch = &vulkan_dispatch_;
        wait_ctx.fingerprint = "vksplat.pipeline.wait_post_submit_fence";
        wait_ctx.owner_quarantine_flag = &gpu_wait_quarantined_;
        auto wait_outcome = lfs::rendering::wait_fence_bounded(
            device,
            fence,
            std::stop_token{},
            lfs::rendering::VulkanWaitPolicy{},
            wait_ctx);
        if (!wait_outcome.has_value() ||
            *wait_outcome != lfs::rendering::WaitOutcome::Ready) {
            PerfTimer::discardMarkers();
            timestampNumWritten = 0;
            PerfTimer::hostTic();
            const std::string outcome_label = wait_outcome.has_value()
                                                  ? std::string(waitOutcomeLabel(*wait_outcome))
                                                  : std::string(wait_outcome.error().detail());
            throwBoundedWaitFailure(
                std::format(
                    "wait_fence_bounded failed after VkSplat submission "
                    "(fence={:#x}, slot={}, outcome={})",
                    lfs::rendering::vkHandleValue(fence),
                    active_command_batch_slot_,
                    outcome_label),
                wait_outcome,
                LFS_SOURCE_SITE_CURRENT());
        }
        const VkResult reset_result = vulkan_dispatch_.reset_fences(device, 1, &fence);
        if (reset_result != VK_SUCCESS) {
            PerfTimer::discardMarkers();
            timestampNumWritten = 0;
            PerfTimer::hostTic();
            lfs::rendering::throw_vk_result(
                reset_result,
                "vkResetFences",
                std::format(
                "vkResetFences failed after retiring a VkSplat batch (fence={:#x}, slot={}, result={}({}))",
                lfs::rendering::vkHandleValue(fence),
                active_command_batch_slot_,
                lfs::rendering::vkResultToString(reset_result),
                static_cast<int>(reset_result)),
                LFS_SOURCE_SITE_CURRENT());
        }
    } else if (signal_semaphore == VK_NULL_HANDLE || signal_value == 0) {
        [[maybe_unused]] auto cpu_timer = timeCpuStage("vksplat.command_batch.wait_idle");
        if (vulkan_dispatch_.queue_wait_idle == nullptr) {
            PerfTimer::discardMarkers();
            timestampNumWritten = 0;
            PerfTimer::hostTic();
            throwRendererContractViolation(
                "endCommandBatch idle path requires VulkanDispatch queue_wait_idle",
                LFS_SOURCE_SITE_CURRENT());
        }
        const VkResult idle_result = vulkan_dispatch_.queue_wait_idle(command_queue);
        if (idle_result != VK_SUCCESS) {
            PerfTimer::discardMarkers();
            timestampNumWritten = 0;
            PerfTimer::hostTic();
            lfs::rendering::throw_vk_result(
                idle_result,
                "vkQueueWaitIdle",
                std::format(
                "vkQueueWaitIdle failed for an unsynchronized VkSplat batch (queue={:#x}, slot={}, result={}({}))",
                lfs::rendering::vkHandleValue(command_queue),
                active_command_batch_slot_,
                lfs::rendering::vkResultToString(idle_result),
                static_cast<int>(idle_result)),
                LFS_SOURCE_SITE_CURRENT());
        }
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
        try {
            collectTimestampResults(slot, timestampNumWritten);
            slot.pending_timestamp_marks.clear();
            timestampNumWritten = 0;
        } catch (...) {
            slot.pending_timestamp_marks.clear();
            timestampNumWritten = 0;
            throw;
        }
    }
}

bool VulkanGSPipeline::writeTimestamp(int delta) {
    if (!commandBatchInProgress) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "writeTimestamp requires an active command batch (batch_active={}, delta={}, written={}, stack_depth={}, command_buffer={:#x})",
            commandBatchInProgress,
            delta,
            timestampNumWritten,
            timestampStackDepth,
            lfs::rendering::vkHandleValue(command_buffer)),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (timestampNumWritten >= MAX_TIMESTAMP_QUERY_COUNT) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "VkSplat timestamp query capacity was exceeded (written={}, capacity={}, delta={}, stack_depth={})",
            timestampNumWritten,
            MAX_TIMESTAMP_QUERY_COUNT,
            delta,
            timestampStackDepth),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (delta != 1 && delta != -1) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "writeTimestamp delta must be +1 or -1 (delta={}, written={}, stack_depth={})",
            delta,
            timestampNumWritten,
            timestampStackDepth),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (delta == -1 && timestampStackDepth == 0) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "writeTimestamp cannot close an empty marker stack (delta={}, written={}, stack_depth={})",
            delta,
            timestampNumWritten,
            timestampStackDepth),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (vulkan_dispatch_.cmd_write_timestamp == nullptr) {
        throwRendererContractViolation(
            "writeTimestamp requires VulkanDispatch::cmd_write_timestamp",
            LFS_SOURCE_SITE_CURRENT());
    }
    vulkan_dispatch_.cmd_write_timestamp(
        command_buffer,
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
    if (vulkan_dispatch_.cmd_write_timestamp == nullptr)
        return false;
    vulkan_dispatch_.cmd_write_timestamp(
        command_buffer,
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
    if (!commandBatchInProgress) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "bufferMemoryBarrier requires an active command batch (batch_active={}, buffer_count={}, dst_mask={}, command_buffer={:#x})",
            commandBatchInProgress,
            buffers.size(),
            static_cast<int>(dstMask),
            lfs::rendering::vkHandleValue(command_buffer)),
            LFS_SOURCE_SITE_CURRENT());
    }

    const VkPipelineStageFlags2 dstStageMask = toStageMask(dstMask);
    const VkAccessFlags2 dstAccessMask = toAccessMask(dstMask);

    std::vector<VkBufferMemoryBarrier2> barriers;
    barriers.reserve(buffers.size());
    for (auto& [buffer, srcMask] : buffers) {
        if (buffer.buffer == VK_NULL_HANDLE) {
            continue;
        }
        validateBufferRange(buffer, 0, buffer.size, "bufferMemoryBarrier");
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

    const VkResult result = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &pipeline.descriptor_set_layout);
    if (result != VK_SUCCESS) {
        lfs::rendering::throw_vk_result(
            result,
            "vkCreateDescriptorSetLayout",
            std::format(
            "VkSplat compute descriptor-set layout creation failed (pipeline='{}', binding_count={}, device={:#x}, result={}({}))",
            pipeline.diagnostic_name,
            bindings.size(),
            lfs::rendering::vkHandleValue(device),
            lfs::rendering::vkResultToString(result),
            static_cast<int>(result)),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (debug_name_writer_.enabled()) {
        setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                           pipeline.descriptor_set_layout,
                           std::format("vksplat.{}.descriptor_layout", pipeline.diagnostic_name));
    }
}

void VulkanGSPipeline::createComputePipeline(_ComputePipeline& pipeline, const std::string& spirv_path, uint32_t min_shared_memory, bool compatible_subgroup_size) {

    if (min_shared_memory > this->deviceInfo.sharedSize) {
        pipeline.shader = VK_NULL_HANDLE;
        return;
    }

    pipeline.diagnostic_name = spirvDiagnosticName(spirv_path);
    const auto spirv_code = loadSpirv(spirv_path);
    recordSlangShaderBytecode(spirv_path, spirv_code.size() * sizeof(uint32_t));
    createShaderModule(spirv_code, &pipeline.shader);
    if (debug_name_writer_.enabled()) {
        setDebugObjectName(VK_OBJECT_TYPE_SHADER_MODULE,
                           pipeline.shader,
                           std::format("vksplat.{}.shader", pipeline.diagnostic_name));
    }
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

    const VkResult layout_result = vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline.pipeline_layout);
    if (layout_result != VK_SUCCESS) {
        lfs::rendering::throw_vk_result(
            layout_result,
            "vkCreatePipelineLayout",
            std::format(
            "VkSplat compute pipeline-layout creation failed (pipeline='{}', descriptor_layout={:#x}, push_constant_bytes={}, result={}({}))",
            pipeline.diagnostic_name,
            lfs::rendering::vkHandleValue(pipeline.descriptor_set_layout),
            push_constant_range.size,
            lfs::rendering::vkResultToString(layout_result),
            static_cast<int>(layout_result)),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (debug_name_writer_.enabled()) {
        setDebugObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                           pipeline.pipeline_layout,
                           std::format("vksplat.{}.pipeline_layout", pipeline.diagnostic_name));
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

    const VkResult pipeline_result =
        vkCreateComputePipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &pipeline.pipeline);
    if (pipeline_result != VK_SUCCESS) {
        lfs::rendering::throw_vk_result(
            pipeline_result,
            "vkCreateComputePipelines",
            std::format(
            "VkSplat compute pipeline creation failed (pipeline='{}', layout={:#x}, shader={:#x}, required_subgroup={}, device_subgroup={}, min_shared_bytes={}, device_shared_bytes={}, result={}({}))",
            pipeline.diagnostic_name,
            lfs::rendering::vkHandleValue(pipeline.pipeline_layout),
            lfs::rendering::vkHandleValue(pipeline.shader),
            compatible_subgroup_size ? SUBGROUP_SIZE : 0,
            deviceInfo.subgroupSize,
            min_shared_memory,
            deviceInfo.sharedSize,
            lfs::rendering::vkResultToString(pipeline_result),
            static_cast<int>(pipeline_result)),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (debug_name_writer_.enabled()) {
        setDebugObjectName(VK_OBJECT_TYPE_PIPELINE,
                           pipeline.pipeline,
                           std::format("vksplat.{}.pipeline", pipeline.diagnostic_name));
    }

    all_compute_pipelines.push_back(&pipeline);
}

void VulkanGSPipeline::executeCompute(
    std::vector<std::pair<size_t, size_t>> dims,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline& pipeline,
    const std::vector<_VulkanBuffer>& buffers) {
    if (uniformSize > MAX_UNIFORM_SIZE || (uniformSize > 0 && uniformsPtr == nullptr)) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "executeCompute push constants require a valid pointer within the VkSplat limit (pipeline='{}', pointer={:#x}, requested_bytes={}, max_bytes={})",
            pipeline.diagnostic_name,
            lfs::rendering::vkHandleValue(uniformsPtr),
            uniformSize,
            MAX_UNIFORM_SIZE),
            LFS_SOURCE_SITE_CURRENT());
    }
    if (pipeline.pipeline == VK_NULL_HANDLE || pipeline.pipeline_layout == VK_NULL_HANDLE ||
        vk_cmd_push_descriptor_set_ == nullptr) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "executeCompute requires a complete compute pipeline (pipeline='{}', pipeline_handle={:#x}, layout={:#x}, push_descriptor_proc={:#x})",
            pipeline.diagnostic_name,
            lfs::rendering::vkHandleValue(pipeline.pipeline),
            lfs::rendering::vkHandleValue(pipeline.pipeline_layout),
            lfs::rendering::vkHandleValue(vk_cmd_push_descriptor_set_)),
            LFS_SOURCE_SITE_CURRENT());
    }

    DEVICE_GUARD;

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

    const std::size_t num_buffers = pipeline.buffer_layouts.size();
    std::vector<VkDescriptorBufferInfo> buffer_infos(num_buffers);
    std::vector<VkWriteDescriptorSet> writes(num_buffers);
    for (std::size_t idx = 0; idx < num_buffers; ++idx) {
        const int binding = pipeline.buffer_layouts[idx];
        if (binding < 0 || static_cast<std::size_t>(binding) >= buffers.size()) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                "executeCompute binding index is outside the supplied buffer array (pipeline='{}', binding={}, buffer_count={}, descriptor_index={})",
                pipeline.diagnostic_name,
                binding,
                buffers.size(),
                idx),
                LFS_SOURCE_SITE_CURRENT());
        }
        const VkDeviceSize range = buffers[binding].size != 0
                                       ? buffers[binding].size
                                       : buffers[binding].capacity;
        validateBufferRange(buffers[binding], 0, range, "executeCompute descriptor binding");
        buffer_infos[idx].buffer = buffers[binding].buffer;
        buffer_infos[idx].offset = buffers[binding].offset;
        // Bind the in-use [offset, offset+size) range. For owned buffers size
        // is set by resizeDeviceBuffer / createBuffer to match the requested
        // allocation; for coalesced views into a parent allocation it's the
        // sub-region's payload byte count. Falling back to the view capacity when size
        // is zero keeps any (rare) legacy callers working without surprises.
        buffer_infos[idx].range = range;

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
    if (dims.empty() || dims.size() > 3) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "executeCompute requires one to three dispatch dimensions (pipeline='{}', dimension_count={})",
            pipeline.diagnostic_name,
            dims.size()),
            LFS_SOURCE_SITE_CURRENT());
    }
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (dims[i].first == 0 || dims[i].second == 0) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                "executeCompute dispatch dimensions must have non-zero element and local sizes (pipeline='{}', dimension={}, elements={}, local_size={})",
                pipeline.diagnostic_name,
                i,
                dims[i].first,
                dims[i].second),
                LFS_SOURCE_SITE_CURRENT());
        }
    }
    while (dims.size() < 3)
        dims.push_back({1, 1});
    uint32_t nGroupsX = (uint32_t)_CEIL_DIV(dims[0].first, dims[0].second);
    uint32_t nGroupsY = (uint32_t)_CEIL_DIV(dims[1].first, dims[1].second);
    uint32_t nGroupsZ = (uint32_t)_CEIL_DIV(dims[2].first, dims[2].second);
    if (nGroupsX > deviceInfo.maxGroupsX ||
        nGroupsY > deviceInfo.maxGroupsY ||
        nGroupsZ > deviceInfo.maxGroupsZ)
        lfs::rendering::throw_renderer_contract(
            std::format(
            "executeCompute dispatch groups exceed the device limit (pipeline='{}', groups=[{},{},{}], maxComputeWorkGroupCount=[{},{},{}], dims=[{}/{},{}/{},{}/{}])",
            pipeline.diagnostic_name,
            nGroupsX,
            nGroupsY,
            nGroupsZ,
            deviceInfo.maxGroupsX,
            deviceInfo.maxGroupsY,
            deviceInfo.maxGroupsZ,
            dims[0].first,
            dims[0].second,
            dims[1].first,
            dims[1].second,
            dims[2].first,
            dims[2].second),
            LFS_SOURCE_SITE_CURRENT());
    vkCmdDispatch(command_buffer, nGroupsX, nGroupsY, nGroupsZ);
}

void VulkanGSPipeline::executeComputeIndirect(
    const _VulkanBuffer& indirect_buffer,
    VkDeviceSize indirect_offset,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline& pipeline,
    const std::vector<_VulkanBuffer>& buffers) {
    if (uniformSize > MAX_UNIFORM_SIZE || (uniformSize > 0 && uniformsPtr == nullptr)) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "executeComputeIndirect push constants require a valid pointer within the VkSplat limit (pipeline='{}', pointer={:#x}, requested_bytes={}, max_bytes={})",
            pipeline.diagnostic_name,
            lfs::rendering::vkHandleValue(uniformsPtr),
            uniformSize,
            MAX_UNIFORM_SIZE),
            LFS_SOURCE_SITE_CURRENT());
    }
    if ((indirect_offset & 3u) != 0) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "executeComputeIndirect requires a four-byte-aligned VkDispatchIndirectCommand offset (pipeline='{}', buffer={:#x}, base_offset={}, relative_offset={}, relative_offset_mod4={})",
            pipeline.diagnostic_name,
            lfs::rendering::vkHandleValue(indirect_buffer.buffer),
            indirect_buffer.offset,
            indirect_offset,
            indirect_offset & 3u),
            LFS_SOURCE_SITE_CURRENT());
    }
    validateBufferRange(indirect_buffer,
                        indirect_offset,
                        sizeof(VkDispatchIndirectCommand),
                        "executeComputeIndirect dispatch arguments");
    if (pipeline.pipeline == VK_NULL_HANDLE || pipeline.pipeline_layout == VK_NULL_HANDLE ||
        vk_cmd_push_descriptor_set_ == nullptr) {
        lfs::rendering::throw_renderer_contract(
            std::format(
            "executeComputeIndirect requires a complete compute pipeline (pipeline='{}', pipeline_handle={:#x}, layout={:#x}, push_descriptor_proc={:#x})",
            pipeline.diagnostic_name,
            lfs::rendering::vkHandleValue(pipeline.pipeline),
            lfs::rendering::vkHandleValue(pipeline.pipeline_layout),
            lfs::rendering::vkHandleValue(vk_cmd_push_descriptor_set_)),
            LFS_SOURCE_SITE_CURRENT());
    }

    DEVICE_GUARD;

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

    const std::size_t num_buffers = pipeline.buffer_layouts.size();
    std::vector<VkDescriptorBufferInfo> buffer_infos(num_buffers);
    std::vector<VkWriteDescriptorSet> writes(num_buffers);
    for (std::size_t idx = 0; idx < num_buffers; ++idx) {
        const int binding = pipeline.buffer_layouts[idx];
        if (binding < 0 || static_cast<std::size_t>(binding) >= buffers.size()) {
            lfs::rendering::throw_renderer_contract(
                std::format(
                "executeComputeIndirect binding index is outside the supplied buffer array (pipeline='{}', binding={}, buffer_count={}, descriptor_index={})",
                pipeline.diagnostic_name,
                binding,
                buffers.size(),
                idx),
                LFS_SOURCE_SITE_CURRENT());
        }
        const VkDeviceSize range = buffers[binding].size != 0
                                       ? buffers[binding].size
                                       : buffers[binding].capacity;
        validateBufferRange(buffers[binding], 0, range, "executeComputeIndirect descriptor binding");
        buffer_infos[idx].buffer = buffers[binding].buffer;
        buffer_infos[idx].offset = buffers[binding].offset;
        // Bind the in-use [offset, offset+size) range. For owned buffers size
        // is set by resizeDeviceBuffer / createBuffer to match the requested
        // allocation; for coalesced views into a parent allocation it's the
        // sub-region's payload byte count. Falling back to the view capacity when size
        // is zero keeps any (rare) legacy callers working without surprises.
        buffer_infos[idx].range = range;

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
