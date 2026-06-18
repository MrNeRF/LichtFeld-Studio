// Compacts the per-chunk touch array the LOD selector wrote into dense
// protected / miss lists so the CPU reads ~K entries instead of scanning
// every logical chunk (851K at 1B splats). Counts overflow instead of
// dropping silently; the touch array rebuilds every frame so an overflow
// costs one frame of tail-priority requests.

struct Uniforms {
    uint32_t chunk_count;
    uint32_t protected_capacity;
    uint32_t miss_capacity;
    uint32_t pad0;
};

layout(binding = 0) StructuredBuffer<uint32_t> chunk_touch;
// [0] protected count, [1] miss count, [2] protected overflow, [3] miss overflow
layout(binding = 1) RWStructuredBuffer<uint32_t> out_counts;
layout(binding = 2) RWStructuredBuffer<uint32_t> out_protected;
// (chunk, priority) pairs
layout(binding = 3) RWStructuredBuffer<uint32_t> out_misses;

static const uint32_t CHUNK_TOUCH_PROTECTED = 0xffffffffu;

[shader("compute")]
    [numthreads(256, 1, 1)] void
    main(uint3 dispatchThreadID
         : SV_DispatchThreadID,
           uniform Uniforms uniforms) {
        const uint32_t chunk = dispatchThreadID.x;
        if (chunk >= uniforms.chunk_count)
            return;
        const uint32_t value = chunk_touch[chunk];
        if (value == 0u)
            return;
        if (value == CHUNK_TOUCH_PROTECTED) {
            uint32_t index = 0u;
            InterlockedAdd(out_counts[0], 1u, index);
            if (index < uniforms.protected_capacity) {
                out_protected[index] = chunk;
            } else {
                uint32_t ignored = 0u;
                InterlockedAdd(out_counts[2], 1u, ignored);
            }
            return;
        }
        uint32_t index = 0u;
        InterlockedAdd(out_counts[1], 1u, index);
        if (index < uniforms.miss_capacity) {
            out_misses[index * 2u + 0u] = chunk;
            out_misses[index * 2u + 1u] = value;
        } else {
            uint32_t ignored = 0u;
            InterlockedAdd(out_counts[3], 1u, ignored);
        }
    }
