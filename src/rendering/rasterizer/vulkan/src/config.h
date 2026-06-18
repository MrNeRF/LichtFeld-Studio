#pragma once

#include <cstdint>

#define ENABLE_VULKAN_VALIDATION_LAYER 0
#define ENABLE_ASSERTION               0

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
// #define SH_REORDER_SIZE 1

typedef int32_t sortingKey_t;
// typedef int64_t sortingKey_t;

#define RASTERIZE_BACKWARD_USE_SCHEDULING 1

#include <cstdio>
#define _DEBUG_PRINT                           \
    do {                                       \
        printf("%s %d\n", __FILE__, __LINE__); \
        fflush(stdout);                        \
    } while (0)
// #define _DEBUG_PRINT ;
#include <cassert>

#include <stdexcept>

#define _THROW_ERROR_ALWAYS(message)                                                          \
    do {                                                                                      \
        std::string msg = std::string(message) +                                              \
                          ". From file `" + __FILE__ + "`, line " + std::to_string(__LINE__); \
        printf("\033[91m%s\033[m\n", msg.c_str());                                            \
        fflush(stdout);                                                                       \
        throw std::runtime_error(msg);                                                        \
    } while (0)

#if ENABLE_ASSERTION
#define _THROW_ERROR(...) _THROW_ERROR_ALWAYS(__VA_ARGS__)
#else
#define _THROW_ERROR(...) \
    do {                  \
    } while (0)
#endif

// Always-on guard, NOT compiled out in release. Use only for conditions that would
// otherwise feed undefined behavior into the Vulkan driver (e.g. a VK_NULL_HANDLE
// buffer reaching vkCmdDispatch). Logic-only invariants stay on _THROW_ERROR.
#define _CHECK_FATAL(...) _THROW_ERROR_ALWAYS(__VA_ARGS__)

#define _CEIL_DIV(x, m)   (((x) + (m) - 1) / (m))
#define _CEIL_ROUND(x, m) (_CEIL_DIV(x, m) * (m))
