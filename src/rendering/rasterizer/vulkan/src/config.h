#pragma once

#include "core/assert.hpp"
#include "rendering/vulkan_result.hpp"

#include <format>
#include <string>

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

#define _THROW_ERROR_ALWAYS(message)                                                          \
    do {                                                                                      \
        std::string msg = std::string(message) +                                              \
                          ". From file `" + __FILE__ + "`, line " + std::to_string(__LINE__); \
        printf("\033[91m%s\033[m\n", msg.c_str());                                            \
        fflush(stdout);                                                                       \
        throw std::runtime_error(msg);                                                        \
    } while (0)

#define _THROW_ERROR(...) _THROW_ERROR_ALWAYS(__VA_ARGS__)

// Vulkan keeps its call-site idiom while sharing the core debug primitive.
// In Release, neither the condition nor the formatting arguments are evaluated.
#ifndef LFS_VK_DEBUG_ASSERT
#define LFS_VK_DEBUG_ASSERT(condition, ...) \
    LFS_DEBUG_ASSERT_MSG(condition, std::format(__VA_ARGS__))
#endif

#define _CEIL_DIV(x, m)   (((x) + (m)-1) / (m))
#define _CEIL_ROUND(x, m) (_CEIL_DIV(x, m) * (m))
