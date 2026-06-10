// Snapshot the depth-ranked slot ids out of the radix ping-pong before the
// tile-instance sort reuses it. Indirect over the visible count — replaces the
// N-sized vkCmdCopyBuffer of the N-wide path.

#include "config.slang"

struct Uniforms {
    uint32_t capacity;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

layout(binding=0) StructuredBuffer<int32_t> sorted_ids;
layout(binding=1) RWStructuredBuffer<int32_t> out_sort_indices;
layout(binding=2) StructuredBuffer<uint32_t> visible_count;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform Uniforms uniforms) {
    const uint idx = dispatchThreadID.x;
    if (idx >= min(visible_count[0], uniforms.capacity))
        return;
    out_sort_indices[idx] = sorted_ids[idx];
}
