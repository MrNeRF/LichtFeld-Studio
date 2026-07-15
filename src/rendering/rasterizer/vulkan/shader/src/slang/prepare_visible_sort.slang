#include "config.slang"
#include "indirect_layout.inc"

struct Uniforms {
    uint32_t num_splats;
    uint32_t sort_partition_size;
    uint32_t pad0;
    uint32_t pad1;
};

layout(binding=0) StructuredBuffer<int32_t> visible_prefix;
layout(binding=1) RWStructuredBuffer<uint32_t> out_visible_count;
layout(binding=2) RWStructuredBuffer<uint32_t> out_sort_dispatch_args;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform Uniforms uniforms) {
    uint32_t visible_count = 0u;
    if (uniforms.num_splats > 0u) {
        int32_t raw_count = visible_prefix[uniforms.num_splats - 1u];
        visible_count = uint32_t(max(raw_count, 0));
        visible_count = min(visible_count, uniforms.num_splats);
    }

    out_visible_count[0] = visible_count;
    out_visible_count[1] = visible_count;

    uint32_t partition_size = max(uniforms.sort_partition_size, 1u);
    uint32_t groups_x = (visible_count + partition_size - 1u) / partition_size;
    const uint32_t radix = LFS_VK_VISIBLE_SORT_DISPATCH_RADIX_WORD_OFFSET;
    out_sort_dispatch_args[radix + LFS_VK_DISPATCH_X_WORD_OFFSET] = max(groups_x, 1u);
    out_sort_dispatch_args[radix + LFS_VK_DISPATCH_Y_WORD_OFFSET] = 1u;
    out_sort_dispatch_args[radix + LFS_VK_DISPATCH_Z_WORD_OFFSET] = 1u;
}
