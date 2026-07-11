#pragma once

#include <cstdint>
#include <format>
#include <string>
#include <type_traits>
#include <vulkan/vulkan.h>

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#define TILE_HEIGHT 16
#define TILE_WIDTH  16

#define RASTER_BATCH_SIZE           1024
#define RASTER_DENSE_TILE_THRESHOLD RASTER_BATCH_SIZE

// HiGS macro-tile inference pipeline (viewer forward only).
// Macro tile = 8x4 render tiles of 8x8 px = 64x32 px. The tile count per
// macro tile must equal SUBGROUP_SIZE: the raster kernel's ballot transpose
// carries one tile per lane.
#define HIGS_MACRO_TILE_WIDTH_TILES  8
#define HIGS_MACRO_TILE_HEIGHT_TILES 4
#define HIGS_MACRO_TILE_SIZE_TILES   (HIGS_MACRO_TILE_WIDTH_TILES * HIGS_MACRO_TILE_HEIGHT_TILES)
#define HIGS_TILE_WIDTH              8
#define HIGS_TILE_HEIGHT             8
#define HIGS_TILE_SIZE               (HIGS_TILE_WIDTH * HIGS_TILE_HEIGHT)
// Macro-tile extent in legacy 16px-tile units (projection rects use that grid).
#define HIGS_MACRO_T16_W ((HIGS_MACRO_TILE_WIDTH_TILES * HIGS_TILE_WIDTH) / TILE_WIDTH)
#define HIGS_MACRO_T16_H ((HIGS_MACRO_TILE_HEIGHT_TILES * HIGS_TILE_HEIGHT) / TILE_HEIGHT)

// Raster/compose run in waves of this many 1024-splat batches so the half4
// partials pool stays bounded (16384 batches x 32 tiles x 256 px x 8 B = 1 GiB).
#define HIGS_RASTER_WAVE_BATCHES 16384
#define HIGS_RASTER_MAX_WAVES    16

// reordering for better memory colaescing
// see config.slang for details
#define SH_REORDER_SIZE SUBGROUP_SIZE

typedef int32_t sortingKey_t;

#include <cassert>
#include <cstdio>

#include <stdexcept>

inline const char* lfsVkResultToString(const VkResult result) noexcept {
    switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
    case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
    case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
    case VK_PIPELINE_COMPILE_REQUIRED: return "VK_PIPELINE_COMPILE_REQUIRED";
    case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
    case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
    case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";
    case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
    case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT: return "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
    case VK_ERROR_NOT_PERMITTED_KHR: return "VK_ERROR_NOT_PERMITTED_KHR";
    case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
    case VK_THREAD_IDLE_KHR: return "VK_THREAD_IDLE_KHR";
    case VK_THREAD_DONE_KHR: return "VK_THREAD_DONE_KHR";
    case VK_OPERATION_DEFERRED_KHR: return "VK_OPERATION_DEFERRED_KHR";
    case VK_OPERATION_NOT_DEFERRED_KHR: return "VK_OPERATION_NOT_DEFERRED_KHR";
    default: return "VK_RESULT_UNKNOWN";
    }
}

template <typename VkHandle>
inline std::uint64_t lfsVkHandleValue(const VkHandle handle) noexcept {
    if constexpr (std::is_pointer_v<VkHandle>) {
        return reinterpret_cast<std::uint64_t>(handle);
    } else {
        return static_cast<std::uint64_t>(handle);
    }
}

#define _THROW_ERROR_ALWAYS(message)                                                          \
    do {                                                                                      \
        std::string msg = std::string(message) +                                              \
                          ". From file `" + __FILE__ + "`, line " + std::to_string(__LINE__); \
        printf("\033[91m%s\033[m\n", msg.c_str());                                            \
        fflush(stdout);                                                                       \
        throw std::runtime_error(msg);                                                        \
    } while (0)

#define _THROW_ERROR(...) _THROW_ERROR_ALWAYS(__VA_ARGS__)

// Retained as a semantic alias at call sites that guard driver-facing handles.
#define _CHECK_FATAL(...) _THROW_ERROR(__VA_ARGS__)

// Hot invariants fail at their first violation in debug builds. Release builds
// do not evaluate either the condition or the formatting arguments.
#ifndef NDEBUG
#define LFS_VK_DEBUG_ASSERT(condition, ...)                \
    do {                                                   \
        if (!(condition)) {                                \
            _THROW_ERROR_ALWAYS(std::format(__VA_ARGS__)); \
        }                                                  \
    } while (false)
#else
#define LFS_VK_DEBUG_ASSERT(condition, ...) ((void)0)
#endif

#define _CEIL_DIV(x, m)   (((x) + (m)-1) / (m))
#define _CEIL_ROUND(x, m) (_CEIL_DIV(x, m) * (m))
