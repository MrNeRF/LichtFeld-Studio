// Two-stage sort: reorders the per-primitive tiles_touched array into
// depth-rank space so the subsequent exclusive cumsum produces offsets that
// match the depth-ordered walk done by generate_keys.
//
// Mirrors gsplat_fwd/kernels_forward.cuh::apply_depth_ordering_cu.

#include "config.slang"

struct Uniforms {
    uint32_t num_splats;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

layout(binding=0) StructuredBuffer<int32_t> primitive_sort_indices;
layout(binding=1) StructuredBuffer<int32_t> tiles_touched_in;
layout(binding=2) RWStructuredBuffer<int32_t> tiles_touched_out;
layout(binding=3) StructuredBuffer<uint32_t> visible_count_in;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint32_t depth_rank = dispatchThreadID.x;
    if (depth_rank >= uniforms.num_splats)
        return;

    uint32_t visible_count = min(visible_count_in[0], uniforms.num_splats);
    if (depth_rank >= visible_count) {
        tiles_touched_out[depth_rank] = 0;
        return;
    }

    int32_t primitive_idx = primitive_sort_indices[depth_rank];
    tiles_touched_out[depth_rank] = tiles_touched_in[primitive_idx];
}
