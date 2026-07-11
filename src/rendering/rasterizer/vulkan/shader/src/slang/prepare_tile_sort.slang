#include "config.slang"
#include "indirect_layout.inc"

// LFS_VISIBLE_BOUNDED=1 (HiGS chain): the cumsum tail lives at
// [visible_count - 1] instead of [num_splats - 1].
#ifndef LFS_VISIBLE_BOUNDED
#define LFS_VISIBLE_BOUNDED 0
#endif

struct Uniforms {
    uint32_t num_splats;
    uint32_t sort_capacity;
    uint32_t sort_partition_size;
    uint32_t pad0;
};

layout(binding=0) StructuredBuffer<int32_t> index_buffer_offset;
layout(binding=1) RWStructuredBuffer<uint32_t> out_tile_instance_count;
layout(binding=2) RWStructuredBuffer<uint32_t> out_sort_dispatch_args;
#if LFS_VISIBLE_BOUNDED
layout(binding=3) StructuredBuffer<uint32_t> visible_count;
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform Uniforms uniforms) {
    uint32_t tile_instance_count = 0u;
    uint32_t raw_instance_count = 0u;
    bool count_overflow = false;
#if LFS_VISIBLE_BOUNDED
    const uint32_t tail = min(visible_count[0], uniforms.num_splats);
    if (tail > 0u) {
        const int32_t signed_count = index_buffer_offset[tail - 1u];
        count_overflow = signed_count < 0;
        if (!count_overflow) {
            raw_instance_count = uint32_t(signed_count);
            tile_instance_count = min(raw_instance_count, uniforms.sort_capacity);
        }
    }
#else
    if (uniforms.num_splats > 0u) {
        const int32_t signed_count = index_buffer_offset[uniforms.num_splats - 1u];
        count_overflow = signed_count < 0;
        if (!count_overflow) {
            raw_instance_count = uint32_t(signed_count);
            tile_instance_count = min(raw_instance_count, uniforms.sort_capacity);
        }
    }
#endif

    out_tile_instance_count[0] = tile_instance_count;
    // [1] keeps the unclamped total so deferred readback can grow capacity.
    // UINT32_MAX cannot be a valid value because the scan is signed int32.
    out_tile_instance_count[1] = count_overflow ? 0xffffffffu : raw_instance_count;

    uint32_t partition_size = max(uniforms.sort_partition_size, 1u);
    uint32_t groups_x = (tile_instance_count + partition_size - 1u) / partition_size;
    const uint32_t radix = LFS_VK_TILE_SORT_DISPATCH_RADIX_WORD_OFFSET;
    out_sort_dispatch_args[radix + LFS_VK_DISPATCH_X_WORD_OFFSET] = max(groups_x, 1u);
    out_sort_dispatch_args[radix + LFS_VK_DISPATCH_Y_WORD_OFFSET] = 1u;
    out_sort_dispatch_args[radix + LFS_VK_DISPATCH_Z_WORD_OFFSET] = 1u;

    const uint32_t range_groups = (tile_instance_count + 1u + 255u) / 256u;
    const uint32_t range = LFS_VK_TILE_SORT_DISPATCH_RANGE_WORD_OFFSET;
    out_sort_dispatch_args[range + LFS_VK_DISPATCH_X_WORD_OFFSET] = max(range_groups, 1u);
    out_sort_dispatch_args[range + LFS_VK_DISPATCH_Y_WORD_OFFSET] = 1u;
    out_sort_dispatch_args[range + LFS_VK_DISPATCH_Z_WORD_OFFSET] = 1u;
}
