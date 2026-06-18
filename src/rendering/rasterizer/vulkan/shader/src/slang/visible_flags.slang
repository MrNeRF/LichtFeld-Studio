#include "config.slang"

struct Uniforms {
    uint32_t num_splats;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

layout(binding=0) StructuredBuffer<int32_t> tiles_touched;
layout(binding=1) RWStructuredBuffer<int32_t> out_visible_flags;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint32_t idx = dispatchThreadID.x;
    if (idx >= uniforms.num_splats)
        return;
    out_visible_flags[idx] = tiles_touched[idx] > 0 ? 1 : 0;
}
