#ifndef LFS_VK_INDIRECT_LAYOUT_INC
#define LFS_VK_INDIRECT_LAYOUT_INC

// Shared word layouts for every rasterizer buffer consumed by
// vkCmdDispatchIndirect. This file is deliberately macro-only so Slang
// producers and C++ allocators/consumers compile from the same constants.

#define LFS_VK_DISPATCH_COMMAND_WORD_COUNT 3u
#define LFS_VK_DISPATCH_X_WORD_OFFSET      0u
#define LFS_VK_DISPATCH_Y_WORD_OFFSET      1u
#define LFS_VK_DISPATCH_Z_WORD_OFFSET      2u

// prepare_visible_sort: one radix-sort dispatch command.
#define LFS_VK_VISIBLE_SORT_DISPATCH_RADIX_WORD_OFFSET 0u
#define LFS_VK_VISIBLE_SORT_DISPATCH_WORD_COUNT        LFS_VK_DISPATCH_COMMAND_WORD_COUNT

// tile_batch_shader descriptors: one raster-batch dispatch command.
#define LFS_VK_TILE_BATCH_DISPATCH_RASTER_WORD_OFFSET 0u
#define LFS_VK_TILE_BATCH_DISPATCH_WORD_COUNT         LFS_VK_DISPATCH_COMMAND_WORD_COUNT

// prepare_visible_chain: radix, generic per-visible work, and two cumsum levels.
#define LFS_VK_VISIBLE_CHAIN_DISPATCH_RADIX_WORD_OFFSET      0u
#define LFS_VK_VISIBLE_CHAIN_DISPATCH_PER_ELEMENT_WORD_OFFSET LFS_VK_DISPATCH_COMMAND_WORD_COUNT
#define LFS_VK_VISIBLE_CHAIN_DISPATCH_CUMSUM_L0_WORD_OFFSET  (2u * LFS_VK_DISPATCH_COMMAND_WORD_COUNT)
#define LFS_VK_VISIBLE_CHAIN_DISPATCH_CUMSUM_L1_WORD_OFFSET  (3u * LFS_VK_DISPATCH_COMMAND_WORD_COUNT)
#define LFS_VK_VISIBLE_CHAIN_DISPATCH_WORD_COUNT             (4u * LFS_VK_DISPATCH_COMMAND_WORD_COUNT)

// cull_splats state: scalar count followed by one projection dispatch command.
#define LFS_VK_SURVIVOR_STATE_COUNT_WORD_OFFSET      0u
#define LFS_VK_SURVIVOR_STATE_PROJECTION_WORD_OFFSET 1u
#define LFS_VK_SURVIVOR_STATE_WORD_COUNT              \
    (LFS_VK_SURVIVOR_STATE_PROJECTION_WORD_OFFSET + LFS_VK_DISPATCH_COMMAND_WORD_COUNT)

// Bounded depth-wave partition output. Slot zero is the frame header; every
// following 256-byte slot is both a storage record and a collection of aligned
// VkDispatchIndirectCommand values for one depth wave.
#define LFS_VK_DEPTH_WAVE_RECORD_STRIDE_WORDS       64u
#define LFS_VK_DEPTH_WAVE_HEADER_NEEDED_WORD         0u
#define LFS_VK_DEPTH_WAVE_COUNT_WORD_OFFSET          0u
#define LFS_VK_DEPTH_WAVE_KEYGEN_WORD_OFFSET         4u
#define LFS_VK_DEPTH_WAVE_RADIX_WORD_OFFSET          7u
#define LFS_VK_DEPTH_WAVE_RANGE_WORD_OFFSET         10u
#define LFS_VK_DEPTH_WAVE_PER_TILE_WORD_OFFSET      13u
#define LFS_VK_DEPTH_WAVE_FULLSCREEN_WORD_OFFSET    16u
#define LFS_VK_DEPTH_WAVE_RANK_BASE_WORD            19u
#define LFS_VK_DEPTH_WAVE_RANK_COUNT_WORD           20u
#define LFS_VK_DEPTH_WAVE_INSTANCE_BASE_WORD        21u

// macro_batch_prepare: one raster and one compose command per wave.
#define LFS_VK_MACRO_WAVE_DISPATCH_WAVE_STRIDE_WORDS LFS_VK_DISPATCH_COMMAND_WORD_COUNT
#define LFS_VK_MACRO_WAVE_DISPATCH_RASTER_BASE_WORD_OFFSET  0u
#define LFS_VK_MACRO_WAVE_DISPATCH_COMPOSE_BASE_WORD_OFFSET \
    (HIGS_RASTER_MAX_WAVES * LFS_VK_MACRO_WAVE_DISPATCH_WAVE_STRIDE_WORDS)
#define LFS_VK_MACRO_WAVE_DISPATCH_WORD_COUNT \
    (2u * HIGS_RASTER_MAX_WAVES * LFS_VK_MACRO_WAVE_DISPATCH_WAVE_STRIDE_WORDS)

#endif // LFS_VK_INDIRECT_LAYOUT_INC
