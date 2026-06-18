// Two-stage sort stage-1 seed: writes identity 0..N-1 into the index payload
// for the radix sort over per-primitive depth keys. Mirrors the role of the
// `primitive_indices` initialization that gsplat_fwd/forward.cu performed
// inline before cub::DeviceRadixSort::SortPairs(depth_keys, primitive_indices).

#include "config.slang"

struct Uniforms {
    uint32_t num_splats;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

layout(binding=0) RWStructuredBuffer<int32_t> out_indices;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint32_t idx = dispatchThreadID.x;
    if (idx >= uniforms.num_splats)
        return;
    out_indices[idx] = int32_t(idx);
}
