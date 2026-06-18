#include "config.slang"

struct Uniforms {
    uint32_t lod_count;
    uint32_t chunk_splats;
    uint32_t invalid_page;
    uint32_t pad0;
};

layout(binding=0) StructuredBuffer<uint32_t> logical_indices;
layout(binding=1) StructuredBuffer<uint32_t> chunk_to_page;
layout(binding=2) RWStructuredBuffer<uint32_t> out_indices;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t idx = dispatchThreadID.x;
    if (idx >= uniforms.lod_count)
        return;

    const uint32_t logical = logical_indices[idx];
    const uint32_t chunk_splats = max(uniforms.chunk_splats, 1u);
    const uint32_t chunk = logical / chunk_splats;
    const uint32_t offset = logical - chunk * chunk_splats;
    const uint32_t page = chunk_to_page[chunk];
    out_indices[idx] = page == uniforms.invalid_page
                           ? uniforms.invalid_page
                           : page * chunk_splats + offset;
}
