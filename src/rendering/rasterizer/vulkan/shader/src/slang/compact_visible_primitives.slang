#include "config.slang"

struct Uniforms {
    uint32_t num_splats;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

layout(binding=0) StructuredBuffer<int32_t> tiles_touched;
layout(binding=1) StructuredBuffer<int32_t> visible_prefix;
layout(binding=2) StructuredBuffer<uint32_t> primitive_depth_keys;

layout(binding=3) RWStructuredBuffer<sortingKey_t> out_compact_depth_keys;
layout(binding=4) RWStructuredBuffer<int32_t> out_compact_indices;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint32_t primitive_idx = dispatchThreadID.x;
    if (primitive_idx >= uniforms.num_splats)
        return;
    if (tiles_touched[primitive_idx] <= 0)
        return;

    int32_t prefix = visible_prefix[primitive_idx];
    if (prefix <= 0)
        return;

    uint32_t compact_idx = uint32_t(prefix - 1);
    out_compact_depth_keys[compact_idx] = sortingKey_t(primitive_depth_keys[primitive_idx]);
    out_compact_indices[compact_idx] = int32_t(primitive_idx);
}
