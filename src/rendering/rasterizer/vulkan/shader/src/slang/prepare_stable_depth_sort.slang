// Export-only preparation for a stable (radial depth, model id) ordering.
// Atomic survivor allocation otherwise makes equal-depth ties scheduling-dependent.
#include "config.slang"

struct Uniforms {
    uint capacity;
    uint restore_depth;
    uint pad0;
    uint pad1;
};

layout(binding=0) StructuredBuffer<uint> input_keys;
layout(binding=1) StructuredBuffer<int> input_ids;
layout(binding=2) RWStructuredBuffer<uint> saved_depth_keys;
layout(binding=3) RWStructuredBuffer<uint> output_keys;
layout(binding=4) RWStructuredBuffer<int> output_ids;
layout(binding=5) StructuredBuffer<uint> visible_count;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform Uniforms uniforms) {
    const uint idx = dispatchThreadID.x;
    if (idx >= min(visible_count[0], uniforms.capacity))
        return;
    if (uniforms.restore_depth != 0u) {
        const int slot = input_ids[idx];
        output_keys[idx] = input_keys[slot];
        output_ids[idx] = slot;
    } else {
        saved_depth_keys[idx] = input_keys[idx];
        output_keys[idx] = uint(input_ids[idx]);
        output_ids[idx] = int(idx);
    }
}
