#include "gs_pipeline.h"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "gs_renderer.h"
#include "perf_timer.h"
#include "shader_variants.hpp"

#include "core/error.hpp"
#include "core/logger.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

static constexpr size_t MAX_UNIFORM_SIZE = sizeof(VulkanGSRendererUniforms);

// The pre-wave renderer fits in the legacy 96-query budget. Each armed depth
// wave adds one independently accumulated cumsum interval (begin + end), and
// PerfTimer deliberately retains every interval so its per-stage count and
// total remain meaningful. Conditional rendering does not predicate timestamp
// writes, so even empty slots consume both queries.
static constexpr uint32_t BASE_TIMESTAMP_QUERY_COUNT = 96;
static constexpr uint32_t DEPTH_WAVE_TIMESTAMP_QUERY_COUNT =
    2u * HIGS_DEPTH_MAX_WAVES;
static constexpr uint32_t MAX_TIMESTAMP_QUERY_COUNT =
    BASE_TIMESTAMP_QUERY_COUNT + DEPTH_WAVE_TIMESTAMP_QUERY_COUNT;

namespace {
    constexpr std::string_view kSlangShaderBytecodeScope = "vksplat.shaders.slang.spirv";
    constexpr std::string_view kSlangShaderRootScope = "vksplat.shaders.slang";
    std::mutex g_spirv_cache_mutex;
    std::unordered_map<std::string, std::shared_ptr<const std::vector<uint32_t>>> g_spirv_cache;

    [[nodiscard]] bool isGeneratedSlangSpirvPath(const std::string& spirv_path) {
        return spirv_path.find("/generated/") != std::string::npos ||
               spirv_path.find("\\generated\\") != std::string::npos;
    }

    [[nodiscard]] std::string spirvDiagnosticName(const std::string& spirv_path) {
        auto name = lfs::core::path_to_utf8(lfs::core::utf8_to_path(spirv_path).stem());
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

    void writeDebugName(const lfs::rendering::VulkanDebugNameWriter& names,
                        const VkObjectType type,
                        const std::uint64_t handle,
                        const std::string_view name) {
        const VkResult result = names.set(type, handle, name);
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

    // What the renderer binds of a compute module: the GLCompute entry point
    // "main", its LocalSize and its descriptor resources. Push constants come
    // only from host code, which the shader compiler checks against its manifest.
    struct SpirvComputeInterface {
        struct Resource {
            uint32_t set = 0;
            uint32_t binding = 0;
            bool storage_buffer = false;
        };
        bool has_main = false;
        std::array<uint32_t, 3> local_size = {}; // zero when sized by LocalSizeId
        std::vector<Resource> resources;
    };

    [[nodiscard]] lfs::Result<SpirvComputeInterface> inspectComputeSpirv(const std::span<const uint32_t> words,
                                                                         const std::string_view label) {
        constexpr uint32_t kMagic = 0x07230203u;
        constexpr uint32_t kMainName[2] = {0x6e69616du, 0u}; // "main\0" as a SPIR-V literal
        if (words.size() < 5 || words[0] != kMagic) {
            return vksplatProgramError(lfs::ErrorCode::InvalidArgument,
                                       std::format("VkSplat compute program '{}' is not SPIR-V (words={}, magic={:#010x})",
                                                   label, words.size(), words.empty() ? 0u : words[0]),
                                       LFS_SOURCE_SITE_CURRENT());
        }
        SpirvComputeInterface result;
        std::unordered_map<uint32_t, uint32_t> pointees; // OpTypePointer id -> pointee type
        std::unordered_map<uint32_t, uint32_t> bindings;
        std::unordered_map<uint32_t, uint32_t> sets;
        std::unordered_set<uint32_t> buffer_blocks;
        struct Variable {
            uint32_t id = 0;
            uint32_t pointer_type = 0;
            uint32_t storage_class = 0;
        };
        std::vector<Variable> variables;
        uint32_t main_id = 0;
        for (size_t word = 5; word < words.size();) {
            const uint32_t count = words[word] >> 16u;
            if (count == 0 || count > words.size() - word) {
                return vksplatProgramError(lfs::ErrorCode::InvalidArgument,
                                           std::format("VkSplat compute program '{}' has a malformed instruction at word {}",
                                                       label, word),
                                           LFS_SOURCE_SITE_CURRENT());
            }
            const auto ins = words.subspan(word, count);
            switch (ins[0] & 0xffffu) {
            case 15: // OpEntryPoint GLCompute %id "main"
                if (count >= 5 && ins[1] == 5u && ins[3] == kMainName[0] && ins[4] == kMainName[1])
                    main_id = ins[2];
                break;
            case 16: // OpExecutionMode %main LocalSize x y z
                if (count >= 6 && main_id != 0 && ins[1] == main_id && ins[2] == 17u)
                    result.local_size = {ins[3], ins[4], ins[5]};
                break;
            case 71: // OpDecorate
                if (count >= 4 && ins[2] == 33u)
                    bindings[ins[1]] = ins[3];
                else if (count >= 4 && ins[2] == 34u)
                    sets[ins[1]] = ins[3];
                else if (count >= 3 && ins[2] == 3u)
                    buffer_blocks.insert(ins[1]);
                break;
            case 32: // OpTypePointer %id StorageClass %type
                if (count >= 4)
                    pointees[ins[1]] = ins[3];
                break;
            case 59: // OpVariable
                if (count >= 4)
                    variables.push_back({ins[2], ins[1], ins[3]});
                break;
            default:
                break;
            }
            word += count;
        }
        result.has_main = main_id != 0;

        constexpr uint32_t kUniform = 2, kStorageBuffer = 12;
        for (const auto& variable : variables) {
            const auto pointer = pointees.find(variable.pointer_type);
            const uint32_t pointee = pointer != pointees.end() ? pointer->second : 0u;
            const auto binding = bindings.find(variable.id);
            const auto set = sets.find(variable.id);
            if (binding == bindings.end() && set == sets.end())
                continue;
            result.resources.push_back({
                .set = set != sets.end() ? set->second : 0u,
                .binding = binding != bindings.end() ? binding->second : 0u,
                .storage_buffer = variable.storage_class == kStorageBuffer ||
                                  (variable.storage_class == kUniform && buffer_blocks.contains(pointee)),
            });
        }
        return result;
    }
} // namespace

lfs::Error vksplatProgramError(const lfs::ErrorCode code, std::string detail, const lfs::core::SourceSite site) {
    return lfs::make_error(lfs::ErrorInit{
        .code = code,
        .domain = lfs::ErrorDomain::Vulkan,
        .user_message = "Shader program setup failed.",
        .detail = std::move(detail),
        .detection = site,
    });
}

[[noreturn]] static void throwRendererContractViolation(std::string detail,
                                                        lfs::core::SourceSite site) {
    // Use the shared typed helper from vulkan_result.hpp.
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

// DeviceLost and other bounded-wait errors preserve their typed Error; flow outcomes map to
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
        code = lfs::ErrorCode::DeadlineExceeded;
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

    {
        const std::lock_guard lock(g_spirv_cache_mutex);
        if (const auto it = g_spirv_cache.find(spirv_path); it != g_spirv_cache.end())
            return *it->second;
    }

    std::ifstream file(lfs::core::native_fs::native_path(lfs::core::native_fs::from_utf8(spirv_path)), std::ios::binary | std::ios::ate);
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

    {
        const std::lock_guard lock(g_spirv_cache_mutex);
        g_spirv_cache[spirv_path] = std::make_shared<const std::vector<uint32_t>>(spirv_code);
    }
    return spirv_code;
}

void preloadSpirvFiles(const std::vector<std::string>& paths) {
    for (const auto& path : paths)
        (void)loadSpirv(path);
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

const lfs::rendering::SubmissionState& VulkanGSPipeline::lastSubmissionState() const noexcept {
    return last_submission_state_;
}

void VulkanGSPipeline::trackExternalParent(VkBuffer buffer) {
    barrier_planner_.track(buffer);
}

void VulkanGSPipeline::untrackExternalParent(VkBuffer buffer) {
    barrier_planner_.forget(buffer);
}

lfs::rendering::vulkan::BufferBarrierPlanner& VulkanGSPipeline::barrierPlanner() noexcept {
    return barrier_planner_;
}

const lfs::rendering::vulkan::BufferBarrierPlanner& VulkanGSPipeline::barrierPlanner() const noexcept {
    return barrier_planner_;
}

void VulkanGSPipeline::planTransfer(
    std::span<const lfs::rendering::vulkan::DeclaredAccess> accesses) {
    if (!commandBatchInProgress) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "planTransfer requires an active command batch (batch_active={}, access_count={}, command_buffer={:#x})",
                commandBatchInProgress,
                accesses.size(),
                lfs::rendering::vkHandleValue(command_buffer)),
            LFS_SOURCE_SITE_CURRENT());
    }
    const auto planned = barrier_planner_.plan(accesses);
    emitPlannedBufferBarriers(planned);
#ifndef NDEBUG
    const int barrier2_emissions = planned.empty() ? 0 : 1;
    assert(barrier2_emissions <= 1);
#endif
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
    // Epic #1496: planner queue family must match barrier emission shape.
    barrier_planner_ = lfs::rendering::vulkan::BufferBarrierPlanner(external_queue_family_index);

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
    createBufferRetireTimeline();

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
    _(xy_vs)
    _(depths)
    _(inv_cov_vs_opacity)
    _(rgb)
    _(custom_geometry)
    _(overlay_flags)
    _(primitive_depth_keys)
    _(primitive_sort_indices)
    _(tiles_touched_depth_ordered)
    _(visible_flags)
    _(visible_block_counts)
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
    _(exact_depth_sample_mask)
    _(macro_wave_args)
    _(index_buffer_offset)
    _(sorting_keys_1)
    _(sorting_keys_2)
    _(sorting_gauss_idx_1)
    _(sorting_gauss_idx_2)
    _(tile_sort_count)
    _(depth_wave_dispatch)
    _(wave_predicates)
    _(tile_ranges)
    _(tile_batch_counts)
    _(tile_batch_offsets)
    _(tile_batch_dispatch_args)
    _(tile_batch_descriptors)
    _(tile_batch_pixel_state)
    _(tile_batch_n_contributors)
    _(pixel_state)
    _(pixel_depth)
    _(pixel_depth_weight)
    _(outline_tiles)
    _(outline_seeds)
    _(outline_nearest)
    _(outline_background)
    _(pixel_outline_ids)
    _(pixel_outline_scene_depth)
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

void VulkanGSPipeline::createBufferRetireTimeline() {
    if (device == VK_NULL_HANDLE) {
        lfs::rendering::throw_renderer_contract(
            "createBufferRetireTimeline requires an initialized device",
            LFS_SOURCE_SITE_CURRENT());
    }
    if (buffer_retire_timeline_ != VK_NULL_HANDLE) {
        return;
    }
    VkSemaphoreTypeCreateInfo type_info{};
    type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type_info.initialValue = 0;

    VkSemaphoreCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    create_info.pNext = &type_info;

    const VkResult result =
        vkCreateSemaphore(device, &create_info, nullptr, &buffer_retire_timeline_);
    if (result != VK_SUCCESS) {
        buffer_retire_timeline_ = VK_NULL_HANDLE;
        lfs::rendering::throw_vk_result(
            result,
            "vkCreateSemaphore",
            std::format(
                "VkSplat buffer-retire timeline creation failed (device={:#x}, result={}({}))",
                lfs::rendering::vkHandleValue(device),
                lfs::rendering::vkResultToString(result),
                static_cast<int>(result)),
            LFS_SOURCE_SITE_CURRENT());
    }
    next_buffer_retire_value_ = 1;
    setDebugObjectName(VK_OBJECT_TYPE_SEMAPHORE,
                       buffer_retire_timeline_,
                       "vksplat.buffer_retire_timeline");
}

void VulkanGSPipeline::destroyBufferRetireTimeline() {
    if (buffer_retire_timeline_ == VK_NULL_HANDLE) {
        return;
    }
    if (device != VK_NULL_HANDLE) {
        vkDestroySemaphore(device, buffer_retire_timeline_, nullptr);
    }
    buffer_retire_timeline_ = VK_NULL_HANDLE;
    next_buffer_retire_value_ = 1;
}

void VulkanGSPipeline::cleanupBuffers(VulkanGSPipelineBuffers& buffers) {
    HOST_GUARD;
    waitForPendingBatch();
    drainRetiredBufferShells(/*force=*/true);
    assert(retired_buffer_shells_.empty());
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
    _(xy_vs)
    _(depths)
    _(inv_cov_vs_opacity)
    _(rgb)
    _(custom_geometry)
    _(overlay_flags)
    _(primitive_depth_keys)
    _(primitive_sort_indices)
    _(tiles_touched_depth_ordered)
    _(visible_flags)
    _(visible_block_counts)
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
    _(exact_depth_sample_mask)
    _(macro_wave_args)
    _(index_buffer_offset)
    _(sorting_keys_1)
    _(sorting_keys_2)
    _(sorting_gauss_idx_1)
    _(sorting_gauss_idx_2)
    _(tile_sort_count)
    _(depth_wave_dispatch)
    _(wave_predicates)
    _(tile_ranges)
    _(tile_batch_counts)
    _(tile_batch_offsets)
    _(tile_batch_dispatch_args)
    _(tile_batch_descriptors)
    _(tile_batch_pixel_state)
    _(tile_batch_n_contributors)
    _(pixel_state)
    _(pixel_depth)
    _(pixel_depth_weight)
    _(outline_tiles)
    _(outline_seeds)
    _(outline_nearest)
    _(outline_background)
    _(pixel_outline_ids)
    _(pixel_outline_scene_depth)
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
    stopProgramSetWorker();
    if (!commandBatchInProgress) {
        waitForPendingBatch();
    }
    HOST_GUARD;
    lfs::diagnostics::VramProfiler::instance().clearStaticScope(kSlangShaderRootScope);

    if (device != VK_NULL_HANDLE) {
        const VkResult idle_result = lfs::rendering::vk_device_wait_idle_synced(device);
        if (idle_result != VK_SUCCESS) {
            LOG_ERROR("Vulkan: vkDeviceWaitIdle failed during VkSplat pipeline cleanup "
                      "(device={:#x}, result={}({})); continuing resource destruction",
                      lfs::rendering::vkHandleValue(device),
                      lfs::rendering::vkResultToString(idle_result),
                      static_cast<int>(idle_result));
        }

        drainRetiredBufferShells(/*force=*/true);
        assert(retired_buffer_shells_.empty());
        destroyBufferRetireTimeline();

        program_sets_.retired.clear();
        program_sets_.current = nullptr;
        program_sets_.sets.clear();
        ++program_set_serial_;
        custom_fixed_.reset();
        custom_variants_.clear();
        custom_effect_active_ = false;
        for (_ComputePipeline* pipeline : all_compute_pipelines)
            destroyComputePipeline(*pipeline);
        for (_ComputePipeline* pipeline : pending_compute_pipelines)
            destroyComputePipeline(*pipeline);
        all_compute_pipelines.clear();
        pending_compute_pipelines.clear();
        banded_export_pipelines_.clear();
        reveal_pipelines_.clear();
        reveal_initialized_ = false;
        reveal_active_ = false;
        banded_export_initialized_ = false;
        banded_export_active_ = false;

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

    // Device already null (or never initialized): drop any leftover shells without VMA free.
    if (!retired_buffer_shells_.empty()) {
        drainRetiredBufferShells(/*force=*/true);
    }
    buffer_retire_timeline_ = VK_NULL_HANDLE;
    next_buffer_retire_value_ = 1;
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
    VkPhysicalDeviceSubgroupSizeControlProperties sizeControlProperties{};
    sizeControlProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    VkPhysicalDeviceSubgroupProperties subgroupProperties{};
    subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroupProperties.pNext = &sizeControlProperties;
    VkPhysicalDeviceProperties2 deviceProperties2{};
    deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProperties2.pNext = &subgroupProperties;
    vkGetPhysicalDeviceProperties2(selected_physical_device, &deviceProperties2);
    const auto& limits = deviceProperties2.properties.limits;

    deviceInfo = {
        .subgroupSize = subgroupProperties.subgroupSize,
        .sharedSize = limits.maxComputeSharedMemorySize,
        .maxGroupsX = limits.maxComputeWorkGroupCount[0],
        .maxGroupsY = limits.maxComputeWorkGroupCount[1],
        .maxGroupsZ = limits.maxComputeWorkGroupCount[2],
        .maxStorageBufferRange = limits.maxStorageBufferRange,
        .maxWorkGroupSize = {limits.maxComputeWorkGroupSize[0],
                             limits.maxComputeWorkGroupSize[1],
                             limits.maxComputeWorkGroupSize[2]},
        .maxWorkGroupInvocations = limits.maxComputeWorkGroupInvocations,
        .minSubgroupSize = sizeControlProperties.minSubgroupSize,
        .maxSubgroupSize = sizeControlProperties.maxSubgroupSize,
        .requiredSubgroupSizeStages = sizeControlProperties.requiredSubgroupSizeStages,
    };
}

void VulkanGSPipeline::setDebugObjectName(const VkObjectType type,
                                          const std::uint64_t handle,
                                          const std::string_view name) const {
    writeDebugName(debug_name_writer_, type, handle, name);
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
    if (size > deviceInfo.maxStorageBufferRange) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "{} exceeds VkPhysicalDeviceLimits::maxStorageBufferRange (buffer={:#x}, allocation={:#x}, base_offset={}, relative_offset={}, range={}, maxStorageBufferRange={}, view_capacity={}, backing_size={}, label='{}')",
                operation,
                lfs::rendering::vkHandleValue(buffer.buffer),
                lfs::rendering::vkHandleValue(buffer.allocation),
                buffer.offset,
                relative_offset,
                size,
                deviceInfo.maxStorageBufferRange,
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

void VulkanGSPipeline::beginCommandBatch() {
    // Cheap non-blocking poll: free growth shells whose timeline keys completed.
    drainRetiredBufferShells(/*force=*/false);

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

    // Epic #1496 §2.4: planner state matches the reuse barrier just recorded.
    barrier_planner_.onBatchBegin();

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
            // A non-ready bounded wait throws and
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
    std::vector<uint64_t> timestamps(timestamp_count);
    const VkResult result = vulkan_dispatch_.get_query_pool_results(
        device, slot.timestamp_query_pool,
        0, timestamp_count,
        sizeof(uint64_t) * timestamp_count,
        timestamps.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT);
    // Profiling must never add an unbounded wait after batch retirement.
    // Missing timestamps can be dropped without affecting the rendered image.
    if (result == VK_NOT_READY)
        return;
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
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physical_device, &deviceProperties);
    double timestampPeriod = deviceProperties.limits.timestampPeriod;

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
        // A non-ready post-submit fence wait throws without reset because reset
        // implies the slot is free for reuse.
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

void VulkanGSPipeline::writeTimestamp(int delta) {
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
    if (barrierMask == VulkanGSPipeline::TRANSFER_READ)
        result |= VK_ACCESS_2_TRANSFER_READ_BIT;
    if (barrierMask == VulkanGSPipeline::TRANSFER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE)
        result |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE)
        result |= VK_ACCESS_2_SHADER_READ_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE)
        result |= VK_ACCESS_2_SHADER_WRITE_BIT;
    if (barrierMask == VulkanGSPipeline::HOST_READ)
        result |= VK_ACCESS_2_HOST_READ_BIT;
    if (barrierMask == VulkanGSPipeline::INDIRECT_DISPATCH_READ)
        result |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    if (barrierMask == VulkanGSPipeline::CONDITIONAL_RENDERING_READ)
        result |= VK_ACCESS_2_CONDITIONAL_RENDERING_READ_BIT_EXT;
    return result;
}

VkPipelineStageFlags2 toStageMask(VulkanGSPipeline::BarrierMask barrierMask) {
    VkPipelineStageFlags2 result = VK_PIPELINE_STAGE_2_NONE;
    if (barrierMask == VulkanGSPipeline::TRANSFER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE)
        result |= VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE)
        result |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    if (barrierMask == VulkanGSPipeline::HOST_READ)
        result |= VK_PIPELINE_STAGE_2_HOST_BIT;
    if (barrierMask == VulkanGSPipeline::INDIRECT_DISPATCH_READ)
        result |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    if (barrierMask == VulkanGSPipeline::CONDITIONAL_RENDERING_READ)
        result |= VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT;
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
        // Epic #1496 §3.4: legacy barrier invalidates planner state for named buffers.
        barrier_planner_.invalidate(buffer.buffer);
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

    if (vulkan_dispatch_.cmd_pipeline_barrier2 == nullptr) {
        throwRendererContractViolation(
            "bufferMemoryBarrier requires VulkanDispatch::cmd_pipeline_barrier2",
            LFS_SOURCE_SITE_CURRENT());
    }

    const uint32_t barrier_count = static_cast<uint32_t>(barriers.size());
    VkDependencyInfo dependency = {};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.bufferMemoryBarrierCount = barrier_count;
    dependency.pBufferMemoryBarriers = barriers.data();

    vulkan_dispatch_.cmd_pipeline_barrier2(command_buffer, &dependency);
}

void VulkanGSPipeline::bufferMemoryBarrier(const std::vector<BufferBarrier>& requested_barriers) {
    if (!commandBatchInProgress) {
        lfs::rendering::throw_renderer_contract(
            std::format(
                "bufferMemoryBarrier requires an active command batch (batch_active={}, buffer_count={}, command_buffer={:#x})",
                commandBatchInProgress,
                requested_barriers.size(),
                lfs::rendering::vkHandleValue(command_buffer)),
            LFS_SOURCE_SITE_CURRENT());
    }

    std::vector<VkBufferMemoryBarrier2> barriers;
    barriers.reserve(requested_barriers.size());
    for (const auto& requested : requested_barriers) {
        const auto& buffer = requested.buffer;
        if (buffer.buffer == VK_NULL_HANDLE)
            continue;
        // Epic #1496 §3.4: legacy barrier invalidates planner state for named buffers.
        barrier_planner_.invalidate(buffer.buffer);
        validateBufferRange(buffer, 0, buffer.size, "bufferMemoryBarrier");
        barriers.push_back(VkBufferMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = toStageMask(requested.src_mask),
            .srcAccessMask = toAccessMask(requested.src_mask),
            .dstStageMask = toStageMask(requested.dst_mask),
            .dstAccessMask = toAccessMask(requested.dst_mask),
            .srcQueueFamilyIndex = queue_family_index,
            .dstQueueFamilyIndex = queue_family_index,
            .buffer = buffer.buffer,
            .offset = buffer.offset,
            .size = buffer.size,
        });
    }
    if (barriers.empty())
        return;

    if (vulkan_dispatch_.cmd_pipeline_barrier2 == nullptr) {
        throwRendererContractViolation(
            "bufferMemoryBarrier requires VulkanDispatch::cmd_pipeline_barrier2",
            LFS_SOURCE_SITE_CURRENT());
    }

    const VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = 0,
        .memoryBarrierCount = 0,
        .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size()),
        .pBufferMemoryBarriers = barriers.data(),
        .imageMemoryBarrierCount = 0,
        .pImageMemoryBarriers = nullptr,
    };
    vulkan_dispatch_.cmd_pipeline_barrier2(command_buffer, &dependency);
}

// Compute pipeline

VulkanGSPipeline::ComputeBuildContext VulkanGSPipeline::computeBuildContext() const {
    return {.device = device, .limits = deviceInfo, .names = debug_name_writer_, .dispatch = vulkan_dispatch_};
}

lfs::Status VulkanGSPipeline::prepareComputeProgram(const ComputeBuildContext& context,
                                                    _ComputePipeline& program,
                                                    const ComputeProgramSpec& spec,
                                                    const std::span<const uint32_t> spirv) {
    const auto reject = [&](const lfs::ErrorCode code, const std::string_view detail, const lfs::core::SourceSite site) {
        return lfs::Status::failure(
            vksplatProgramError(code, std::format("VkSplat compute program '{}' {}", spec.label, detail), site));
    };
    auto inspected = inspectComputeSpirv(spirv, spec.label);
    if (!inspected)
        return lfs::Status::failure(std::move(inspected).error());
    const auto& shader = *inspected;
    const auto& limits = context.limits;
    if (!shader.has_main)
        return reject(lfs::ErrorCode::InvalidArgument, "has no GLCompute entry point named main",
                      LFS_SOURCE_SITE_CURRENT());
    for (const auto& resource : shader.resources) {
        if (resource.set != 0 || !resource.storage_buffer ||
            std::ranges::find(spec.bindings, static_cast<int>(resource.binding)) == spec.bindings.end()) {
            return reject(lfs::ErrorCode::ContractViolation,
                          std::format("declares set {} binding {} (storage_buffer={}), which the host layout of {} "
                                      "storage buffers in set 0 does not provide",
                                      resource.set, resource.binding, resource.storage_buffer, spec.bindings.size()),
                          LFS_SOURCE_SITE_CURRENT());
        }
    }
    const auto& local = shader.local_size;
    if (spec.workgroup != std::array<uint32_t, 3>{} && local != spec.workgroup) {
        return reject(lfs::ErrorCode::ContractViolation,
                      std::format("declares workgroup [{},{},{}], the host dispatches [{},{},{}]",
                                  local[0], local[1], local[2],
                                  spec.workgroup[0], spec.workgroup[1], spec.workgroup[2]),
                      LFS_SOURCE_SITE_CURRENT());
    }
    if (local[0] != 0 && limits.maxWorkGroupInvocations != 0 &&
        (local[0] > limits.maxWorkGroupSize[0] || local[1] > limits.maxWorkGroupSize[1] ||
         local[2] > limits.maxWorkGroupSize[2] ||
         uint64_t{local[0]} * local[1] * local[2] > limits.maxWorkGroupInvocations)) {
        return reject(lfs::ErrorCode::Unsupported,
                      std::format("workgroup [{},{},{}] exceeds the device limits [{},{},{}] and {} invocations",
                                  local[0], local[1], local[2], limits.maxWorkGroupSize[0],
                                  limits.maxWorkGroupSize[1], limits.maxWorkGroupSize[2],
                                  limits.maxWorkGroupInvocations),
                      LFS_SOURCE_SITE_CURRENT());
    }
    const uint32_t subgroup = spec.required_subgroup_size;
    if (subgroup != 0 && subgroup != limits.subgroupSize && limits.maxSubgroupSize != 0 &&
        (!std::has_single_bit(subgroup) || subgroup < limits.minSubgroupSize || subgroup > limits.maxSubgroupSize ||
         (limits.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0)) {
        return reject(lfs::ErrorCode::Unsupported,
                      std::format("requires subgroup size {}, the device pins {}..{} for compute stages {:#x}",
                                  subgroup, limits.minSubgroupSize, limits.maxSubgroupSize,
                                  limits.requiredSubgroupSizeStages),
                      LFS_SOURCE_SITE_CURRENT());
    }
    _ComputePipeline record(spec.bindings);
    record.diagnostic_name = spec.label;
    record.required_subgroup_size = spec.required_subgroup_size;
    record.expected_workgroup_size_x = spec.workgroup[0];
    const auto& vk = context.dispatch;
    const auto vk_failure = [&](const VkResult result, const std::string_view operation, auto& failed_handle,
                                const lfs::core::SourceSite site) {
        failed_handle = VK_NULL_HANDLE;
        destroyComputeProgram(context.device, vk, record);
        return lfs::Status::failure(lfs::rendering::make_vk_result_error(
            result,
            std::format("{} failed for VkSplat compute program '{}' (device={:#x}, result={}({}))",
                        operation, spec.label, lfs::rendering::vkHandleValue(context.device),
                        lfs::rendering::vkResultToString(result), static_cast<int>(result)),
            site));
    };
    const auto name = [&](const VkObjectType type, const auto handle, const std::string_view suffix) {
        if (context.names.enabled())
            writeDebugName(context.names, type, lfs::rendering::vkHandleValue(handle),
                           std::format("vksplat.{}.{}", spec.label, suffix));
    };

    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = spirv.size_bytes();
    module_info.pCode = spirv.data();
    VkResult result = vk.create_shader_module(context.device, &module_info, nullptr, &record.shader);
    if (result != VK_SUCCESS)
        return vk_failure(result, "vkCreateShaderModule", record.shader, LFS_SOURCE_SITE_CURRENT());
    name(VK_OBJECT_TYPE_SHADER_MODULE, record.shader, "shader");

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(record.buffer_layouts.size());
    for (const int binding : record.buffer_layouts) {
        bindings.push_back({
            .binding = static_cast<uint32_t>(binding),
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        });
    }
    VkDescriptorSetLayoutCreateInfo set_layout_info{};
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    set_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    set_layout_info.pBindings = bindings.data();
    result = vk.create_descriptor_set_layout(context.device, &set_layout_info, nullptr, &record.descriptor_set_layout);
    if (result != VK_SUCCESS) {
        return vk_failure(result, "vkCreateDescriptorSetLayout", record.descriptor_set_layout,
                          LFS_SOURCE_SITE_CURRENT());
    }
    name(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, record.descriptor_set_layout, "descriptor_layout");

    VkPushConstantRange push_constant_range{};
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = static_cast<uint32_t>(MAX_UNIFORM_SIZE);
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &record.descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant_range;
    result = vk.create_pipeline_layout(context.device, &pipeline_layout_info, nullptr, &record.pipeline_layout);
    if (result != VK_SUCCESS)
        return vk_failure(result, "vkCreatePipelineLayout", record.pipeline_layout, LFS_SOURCE_SITE_CURRENT());
    name(VK_OBJECT_TYPE_PIPELINE_LAYOUT, record.pipeline_layout, "pipeline_layout");

    program = std::move(record);
    return {};
}

lfs::Status VulkanGSPipeline::createComputePrograms(const ComputeBuildContext& context,
                                                    const VkPipelineCache cache,
                                                    const std::span<_ComputePipeline* const> programs) {
    if (programs.empty())
        return {};

    std::vector<VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT> subgroup_infos;
    subgroup_infos.reserve(programs.size());
    std::vector<VkComputePipelineCreateInfo> infos;
    infos.reserve(programs.size());
    std::vector<VkPipeline> pipelines(programs.size(), VK_NULL_HANDLE);

    for (auto* const program : programs) {
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_info{};
        subgroup_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
        subgroup_info.requiredSubgroupSize = program->required_subgroup_size;
        subgroup_infos.push_back(subgroup_info);

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = program->shader;
        stage.pName = "main";
        if (program->required_subgroup_size != 0 && context.limits.subgroupSize != program->required_subgroup_size)
            stage.pNext = &subgroup_infos.back();

        VkComputePipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.layout = program->pipeline_layout;
        info.stage = stage;
        infos.push_back(info);
    }

    const VkResult result = context.dispatch.create_compute_pipelines(
        context.device, cache, static_cast<uint32_t>(infos.size()), infos.data(), nullptr, pipelines.data());
    if (result != VK_SUCCESS) {
        std::string_view failed;
        for (std::size_t i = 0; i < pipelines.size(); ++i) {
            if (pipelines[i] != VK_NULL_HANDLE)
                context.dispatch.destroy_pipeline(context.device, pipelines[i], nullptr);
            else if (failed.empty())
                failed = programs[i]->diagnostic_name;
        }
        return lfs::Status::failure(lfs::rendering::make_vk_result_error(
            result,
            std::format("VkSplat batched compute pipeline creation failed (count={}, first_failed='{}', result={}({}))",
                        infos.size(), failed, lfs::rendering::vkResultToString(result), static_cast<int>(result)),
            LFS_SOURCE_SITE_CURRENT()));
    }

    for (std::size_t i = 0; i < programs.size(); ++i) {
        auto& program = *programs[i];
        program.pipeline = pipelines[i];
        if (context.names.enabled()) {
            writeDebugName(context.names, VK_OBJECT_TYPE_PIPELINE, lfs::rendering::vkHandleValue(program.pipeline),
                           std::format("vksplat.{}.pipeline", program.diagnostic_name));
        }
    }
    return {};
}

void VulkanGSPipeline::destroyComputeProgram(const VkDevice device,
                                             const lfs::rendering::VulkanDispatch& dispatch,
                                             _ComputePipeline& program) noexcept {
    if (program.descriptor_set_layout != VK_NULL_HANDLE) {
        dispatch.destroy_descriptor_set_layout(device, program.descriptor_set_layout, nullptr);
        program.descriptor_set_layout = VK_NULL_HANDLE;
    }
    if (program.pipeline != VK_NULL_HANDLE) {
        dispatch.destroy_pipeline(device, program.pipeline, nullptr);
        program.pipeline = VK_NULL_HANDLE;
    }
    if (program.pipeline_layout != VK_NULL_HANDLE) {
        dispatch.destroy_pipeline_layout(device, program.pipeline_layout, nullptr);
        program.pipeline_layout = VK_NULL_HANDLE;
    }
    if (program.shader != VK_NULL_HANDLE) {
        dispatch.destroy_shader_module(device, program.shader, nullptr);
        program.shader = VK_NULL_HANDLE;
    }
}

lfs::Status VulkanGSPipeline::createComputePipeline(_ComputePipeline& pipeline,
                                                    const ComputeProgramSpec& spec,
                                                    const std::span<const uint32_t> spirv) {
    if (auto prepared = prepareComputeProgram(computeBuildContext(), pipeline, spec, spirv); !prepared)
        return prepared;
    pending_compute_pipelines.push_back(&pipeline);
    return {};
}

void VulkanGSPipeline::createComputePipeline(_ComputePipeline& pipeline,
                                             const std::string& spirv_path,
                                             const bool compatible_subgroup_size,
                                             const uint32_t expected_workgroup_size_x) {
    const auto spirv_code = loadSpirv(spirv_path);
    const uint32_t checked = expected_workgroup_size_x != 0 ? 1u : 0u;
    const ComputeProgramSpec spec{
        .label = spirvDiagnosticName(spirv_path),
        .bindings = pipeline.buffer_layouts,
        .workgroup = {expected_workgroup_size_x, checked, checked},
        .required_subgroup_size = compatible_subgroup_size ? uint32_t{SUBGROUP_SIZE} : 0u,
    };
    if (auto created = createComputePipeline(pipeline, spec, spirv_code); !created) {
        lfs::rendering::throw_vk_error(
            std::move(created).error().with_context(std::format("load '{}'", spirv_path), LFS_SOURCE_SITE_CURRENT()));
    }
    pipeline.spirv_path = spirv_path;
    recordSlangShaderBytecode(spirv_path, spirv_code.size() * sizeof(uint32_t));
}

void VulkanGSPipeline::initializeLegacyVariants(const bool reveal) {
    const size_t count = all_compute_pipelines.size();
    const size_t pending_count = pending_compute_pipelines.size();
    const size_t banded_count = banded_export_pipelines_.size();
    const size_t reveal_count = reveal_pipelines_.size();
    auto& variants = reveal ? reveal_pipelines_ : banded_export_pipelines_;
    const auto append = [&](auto& collection, _ComputePipeline* base, const auto& path, const std::string_view label) {
        auto variant = std::make_unique<_ComputePipeline>(base->buffer_layouts);
        auto* result = variant.get();
        collection.emplace_back(base, std::move(variant));
        createComputePipeline(*result, lfs::core::path_to_utf8(path / (std::string(label) + ".spv")),
                              base->required_subgroup_size != 0, base->expected_workgroup_size_x);
        return result;
    };
    try {
        for (size_t i = 0; i < count; ++i) {
            auto* const pipeline = all_compute_pipelines[i];
            const std::filesystem::path path(pipeline->spirv_path);
            const auto* row = lfs::rendering::shaderVariants(lfs::core::path_to_utf8(path.stem()));
            const auto label = row ? (reveal ? row->reveal : row->banded) : std::string_view{};
            if (label.empty())
                continue;
            auto* variant = append(variants, pipeline, path.parent_path(), label);
            if (reveal && banded_export_initialized_) {
                const auto* reveal_row = lfs::rendering::shaderVariants(label);
                if (reveal_row && !reveal_row->banded.empty())
                    append(banded_export_pipelines_, variant, path.parent_path(), reveal_row->banded);
            }
        }
        createPendingComputePipelines();
    } catch (...) {
        for (size_t i = banded_count; i < banded_export_pipelines_.size(); ++i)
            destroyComputePipeline(*banded_export_pipelines_[i].second);
        for (size_t i = reveal_count; i < reveal_pipelines_.size(); ++i)
            destroyComputePipeline(*reveal_pipelines_[i].second);
        all_compute_pipelines.resize(count);
        pending_compute_pipelines.resize(pending_count);
        banded_export_pipelines_.resize(banded_count);
        reveal_pipelines_.resize(reveal_count);
        throw;
    }
}

void VulkanGSPipeline::setRevealEffect(const bool enabled) {
    if (enabled == reveal_active_)
        return;
    const bool banded = banded_export_active_;
    if (reveal_active_) {
        for (auto& [pipeline, variant] : reveal_pipelines_)
            pipeline->swapProgram(*variant);
        reveal_active_ = false;
    }
    if (banded)
        setBandedExport(false);
    if (enabled && !reveal_initialized_) {
        try {
            initializeLegacyVariants(true);
        } catch (...) {
            if (banded)
                setBandedExport(true);
            throw;
        }
        reveal_initialized_ = true;
    }
    if (banded)
        setBandedExport(true);
    if (enabled) {
        for (auto& [pipeline, variant] : reveal_pipelines_)
            pipeline->swapProgram(*variant);
    }
    reveal_active_ = enabled;
}

void VulkanGSPipeline::setBandedExport(const bool enabled) {
    if (enabled == banded_export_active_)
        return;
    if (reveal_active_) {
        setRevealEffect(false);
        setBandedExport(enabled);
        setRevealEffect(true);
        return;
    }
    if (enabled && !banded_export_initialized_) {
        initializeLegacyVariants(false);
        banded_export_initialized_ = true;
    }
    // Swap once at the boundary; all variants survive in-flight uses until teardown.
    for (auto& [pipeline, variant] : banded_export_pipelines_)
        pipeline->swapProgram(*variant);
    banded_export_active_ = enabled;
}

void VulkanGSPipeline::createPendingComputePipelines() {
    if (auto created = createComputePrograms(computeBuildContext(), pipeline_cache, pending_compute_pipelines);
        !created) {
        for (auto* const pipeline : pending_compute_pipelines)
            destroyComputePipeline(*pipeline);
        pending_compute_pipelines.clear();
        lfs::rendering::throw_vk_error(std::move(created).error());
    }
    all_compute_pipelines.insert(all_compute_pipelines.end(),
                                 pending_compute_pipelines.begin(), pending_compute_pipelines.end());
    pending_compute_pipelines.clear();
}

void VulkanGSPipeline::emitPlannedBufferBarriers(
    const std::vector<VkBufferMemoryBarrier2>& barriers) {
    if (barriers.empty()) {
        return;
    }
    if (vulkan_dispatch_.cmd_pipeline_barrier2 == nullptr) {
        throwRendererContractViolation(
            "tagged dispatch requires VulkanDispatch::cmd_pipeline_barrier2",
            LFS_SOURCE_SITE_CURRENT());
    }
    const VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = 0,
        .memoryBarrierCount = 0,
        .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size()),
        .pBufferMemoryBarriers = barriers.data(),
        .imageMemoryBarrierCount = 0,
        .pImageMemoryBarriers = nullptr,
    };
    // Spec §2.6 / §3.1: at most one barrier2 call per tagged dispatch (coalescing).
    vulkan_dispatch_.cmd_pipeline_barrier2(command_buffer, &dependency);
}

void VulkanGSPipeline::recordComputeDispatch(
    std::vector<std::pair<size_t, size_t>> dims,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline& pipeline,
    const std::vector<_VulkanBuffer>& buffers) {
    if (vulkan_dispatch_.cmd_bind_pipeline == nullptr ||
        vulkan_dispatch_.cmd_push_constants == nullptr ||
        vulkan_dispatch_.cmd_dispatch == nullptr) {
        throwRendererContractViolation(
            "executeCompute requires VulkanDispatch cmd_bind_pipeline, "
            "cmd_push_constants, and cmd_dispatch",
            LFS_SOURCE_SITE_CURRENT());
    }

    vulkan_dispatch_.cmd_bind_pipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

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

    if (uniformsPtr) {
        vulkan_dispatch_.cmd_push_constants(
            command_buffer,
            pipeline.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, (uint32_t)uniformSize, uniformsPtr);
    }

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
    vulkan_dispatch_.cmd_dispatch(command_buffer, nGroupsX, nGroupsY, nGroupsZ);
}

void VulkanGSPipeline::recordComputeDispatchIndirect(
    const _VulkanBuffer& indirect_buffer,
    VkDeviceSize indirect_offset,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline& pipeline,
    const std::vector<_VulkanBuffer>& buffers) {
    if (vulkan_dispatch_.cmd_bind_pipeline == nullptr ||
        vulkan_dispatch_.cmd_push_constants == nullptr ||
        vulkan_dispatch_.cmd_dispatch_indirect == nullptr) {
        throwRendererContractViolation(
            "executeComputeIndirect requires VulkanDispatch cmd_bind_pipeline, "
            "cmd_push_constants, and cmd_dispatch_indirect",
            LFS_SOURCE_SITE_CURRENT());
    }

    vulkan_dispatch_.cmd_bind_pipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

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
        vulkan_dispatch_.cmd_push_constants(
            command_buffer,
            pipeline.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, (uint32_t)uniformSize, uniformsPtr);
    }

    vulkan_dispatch_.cmd_dispatch_indirect(
        command_buffer, indirect_buffer.buffer, indirect_buffer.offset + indirect_offset);
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

    // Epic #1496 §3.4: untagged path invalidates every bound buffer (mixed-mode safety).
    for (const auto& buffer : buffers) {
        barrier_planner_.invalidate(buffer.buffer);
    }

    DEVICE_GUARD;
    recordComputeDispatch(std::move(dims), uniformsPtr, uniformSize, pipeline, buffers);
}

void VulkanGSPipeline::executeCompute(
    std::vector<std::pair<size_t, size_t>> dims,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline& pipeline,
    const std::vector<TaggedBinding>& bindings) {
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

    // Dense ordered list: same indexing as the untagged buffer array.
    std::vector<_VulkanBuffer> buffers;
    buffers.reserve(bindings.size());
    for (const auto& binding : bindings) {
        buffers.push_back(binding.buffer);
    }

    std::vector<lfs::rendering::vulkan::DeclaredAccess> accesses;
    accesses.reserve(bindings.size());
    for (std::size_t i = 0; i < bindings.size(); ++i) {
        accesses.push_back(lfs::rendering::vulkan::DeclaredAccess{
            .buffer = &buffers[i],
            .use = bindings[i].use,
        });
    }

    DEVICE_GUARD;

    const auto planned = barrier_planner_.plan(accesses);
    // Coalescing: one barrier2 for the whole planned set (0 or 1 emission).
    emitPlannedBufferBarriers(planned);
#ifndef NDEBUG
    // Spec §2.6: each tagged dispatch emits ≤ 1 vkCmdPipelineBarrier2 call.
    const int barrier2_emissions = planned.empty() ? 0 : 1;
    assert(barrier2_emissions <= 1);
#endif

    recordComputeDispatch(std::move(dims), uniformsPtr, uniformSize, pipeline, buffers);
}

void VulkanGSPipeline::executeComputeIndirect(
    const _VulkanBuffer& indirect_buffer,
    VkDeviceSize indirect_offset,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline& pipeline,
    const std::vector<TaggedBinding>& bindings) {
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

    std::vector<_VulkanBuffer> buffers;
    buffers.reserve(bindings.size());
    for (const auto& binding : bindings) {
        buffers.push_back(binding.buffer);
    }

    std::vector<lfs::rendering::vulkan::DeclaredAccess> accesses;
    accesses.reserve(bindings.size() + 1);
    for (std::size_t i = 0; i < bindings.size(); ++i) {
        accesses.push_back(lfs::rendering::vulkan::DeclaredAccess{
            .buffer = &buffers[i],
            .use = bindings[i].use,
        });
    }
    // Implicit IndirectRead on the dispatch-args buffer (spec §3.1).
    accesses.push_back(lfs::rendering::vulkan::DeclaredAccess{
        .buffer = &indirect_buffer,
        .use = lfs::rendering::vulkan::BufferUse::IndirectRead,
    });

    DEVICE_GUARD;

    const auto planned = barrier_planner_.plan(accesses);
    emitPlannedBufferBarriers(planned);
#ifndef NDEBUG
    const int barrier2_emissions = planned.empty() ? 0 : 1;
    assert(barrier2_emissions <= 1);
#endif

    recordComputeDispatchIndirect(
        indirect_buffer, indirect_offset, uniformsPtr, uniformSize, pipeline, buffers);
}

void VulkanGSPipeline::destroyComputePipeline(_ComputePipeline& pipeline) {
    destroyComputeProgram(device, vulkan_dispatch_, pipeline);
}
