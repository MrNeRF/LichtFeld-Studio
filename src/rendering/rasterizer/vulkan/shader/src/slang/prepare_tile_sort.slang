// LFS_VISIBLE_BOUNDED=1 (HiGS chain): the cumsum tail lives at
// [visible_count - 1] instead of [num_splats - 1].
#ifndef LFS_VISIBLE_BOUNDED
#define LFS_VISIBLE_BOUNDED 0
#endif

struct Uniforms {
    uint32_t num_splats;
};

layout(binding=0) StructuredBuffer<int32_t> index_buffer_offset;
layout(binding=1) RWStructuredBuffer<uint32_t> out_tile_instance_count;
#if LFS_VISIBLE_BOUNDED
layout(binding=2) StructuredBuffer<uint32_t> visible_count;
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform Uniforms uniforms) {
    uint32_t raw_instance_count = 0u;
    bool count_overflow = false;
#if LFS_VISIBLE_BOUNDED
    const uint32_t tail = min(visible_count[0], uniforms.num_splats);
    if (tail > 0u) {
        const int32_t signed_count = index_buffer_offset[tail - 1u];
        count_overflow = signed_count < 0 || signed_count >= int32_t(0x7fffffffu);
        if (!count_overflow)
            raw_instance_count = uint32_t(signed_count);
    }
#else
    if (uniforms.num_splats > 0u) {
        const int32_t signed_count = index_buffer_offset[uniforms.num_splats - 1u];
        count_overflow = signed_count < 0 || signed_count >= int32_t(0x7fffffffu);
        if (!count_overflow)
            raw_instance_count = uint32_t(signed_count);
    }
#endif

    // The raw total feeds telemetry, the export gate, and loud overflow detection.
    // UINT32_MAX cannot be a valid value because the scan is signed int32.
    out_tile_instance_count[0] = count_overflow ? 0xffffffffu : raw_instance_count;
}
