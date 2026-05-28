#include "config.slang"

struct Uniforms {
    uint32_t num_splats;
    uint32_t sort_capacity;
    uint32_t sort_partition_size;
    uint32_t pad0;
};

layout(binding=0) StructuredBuffer<int32_t> index_buffer_offset;
layout(binding=1) RWStructuredBuffer<uint32_t> out_tile_instance_count;
layout(binding=2) RWStructuredBuffer<uint32_t> out_sort_dispatch_args;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform Uniforms uniforms) {
    uint32_t tile_instance_count = 0u;
    if (uniforms.num_splats > 0u) {
        int32_t raw_count = index_buffer_offset[uniforms.num_splats - 1u];
        tile_instance_count = uint32_t(max(raw_count, 0));
        tile_instance_count = min(tile_instance_count, uniforms.sort_capacity);
    }

    out_tile_instance_count[0] = tile_instance_count;

    uint32_t partition_size = max(uniforms.sort_partition_size, 1u);
    uint32_t groups_x = (tile_instance_count + partition_size - 1u) / partition_size;
    out_sort_dispatch_args[0] = max(groups_x, 1u);
    out_sort_dispatch_args[1] = 1u;
    out_sort_dispatch_args[2] = 1u;
}
