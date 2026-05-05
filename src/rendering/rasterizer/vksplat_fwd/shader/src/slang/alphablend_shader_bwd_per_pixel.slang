#define SUBGROUP_REDUCE_BEFORE_ATOMIC 1
#define SKIP_ZERO_IN_ATOMIC 1


#if SHARED_STRUCT
groupshared Splat_2D_AlphaBlend collected_splats[TILE_SIZE];
#else
groupshared float2 collected_xy_vs[TILE_SIZE];
groupshared float4 collected_inv_cov_vs_opacity[TILE_SIZE];
#endif

groupshared uint32_t collected_idx[TILE_SIZE];

groupshared float subgroup_shared[TILE_SIZE];


layout(binding=0) StructuredBuffer<int32_t> sorted_gauss_idx;
layout(binding=1) StructuredBuffer<int32_t> tile_ranges;
layout(binding=2) StructuredBuffer<float2> xy_vs;
layout(binding=3) StructuredBuffer<float4> inv_cov_vs_opacity;
layout(binding=4) StructuredBuffer<float> rgb;
layout(binding=5) StructuredBuffer<float4> final_pixel_state;
layout(binding=6) StructuredBuffer<int32_t> n_contributors;
layout(binding=7) StructuredBuffer<float4> v_current_pixel_state;

layout(binding=8) RWByteAddressBuffer v_xy_vs;
layout(binding=9) RWByteAddressBuffer v_inv_cov_vs_opacity;
layout(binding=10) RWByteAddressBuffer v_rgb;

[shader("compute")]
[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform Uniforms uniforms
) {
    const uint32_t H = uniforms.image_height;
    const uint32_t W = uniforms.image_width;
    const uint grid_width = uniforms.grid_width;

    uint2 pix_coord = dispatchThreadID.xy;

    uint tile_idx = groupID.y * grid_width + groupID.x;
    uint32_t tile_idx_start = tile_ranges[tile_idx];
    uint32_t tile_idx_end = tile_ranges[tile_idx+1];

    uint32_t thread_rank = groupThreadID.y * TILE_WIDTH + groupThreadID.x;

    float4 d_current_pixel_state;

    // Load the final pixel state.
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    uint32_t block_size = TILE_HEIGHT * TILE_WIDTH;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);

    int splats_left_to_process = tile_idx_end - tile_idx_start;
    uint32_t current_splat_offset = tile_idx_end - tile_idx_start;

    float4 current_pixel_state;
    int32_t n_contrib_fwd;
    if (is_inside) {
        uint32_t pix_id = pix_coord.y * W + pix_coord.x;
        current_pixel_state = final_pixel_state[pix_id];
        n_contrib_fwd = n_contributors[pix_id];
        d_current_pixel_state = v_current_pixel_state[pix_id];
    }

    float2 center_pix_coord = pix_coord;

    for (int i = 0; i < rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * block_size + thread_rank;
        if (tile_idx_start + progress < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_end - progress - 1]);
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_splat_alphablend(coll_id, xy_vs, inv_cov_vs_opacity, rgb);
        }
        GroupMemoryBarrierWithGroupSync();

        for (int j = 0; j < min(block_size, splats_left_to_process); j++)
        {
            uint32_t g_idx = collected_idx[j];
            DifferentialPair<Splat_2D_AlphaBlend> dp_g;
            #if EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD
                float2 d_xy_vs = float2(0.0);
                float4 d_inv_cov_vs_opacity = float4(0.0);
                float3 d_rgb = float3(0.0);
            #endif

            if (is_inside) {
                current_splat_offset--;
                if (current_splat_offset < n_contrib_fwd) {

                Splat_2D_AlphaBlend g = collected_splats[j];

                float4 gauss_rgba = evaluate_splat(g, center_pix_coord);
                if (gauss_rgba.a >= ALPHA_THRESHOLD) {

                // Undo pixel state
                current_pixel_state = undo_pixel_state(current_pixel_state, gauss_rgba);

                // Back-prop automatically through blending and gaussian evaluation.
                dp_g = diffPair(g);
                DifferentialPair<float4> dp_gauss_rgba = diffPair(gauss_rgba);
                DifferentialPair<float4> dp_current_pixel_state = diffPair(current_pixel_state);

                bwd_diff(update_pixel_state)(dp_current_pixel_state, dp_gauss_rgba, d_current_pixel_state);
                d_current_pixel_state = dp_current_pixel_state.getDifferential();
                bwd_diff(evaluate_splat)(dp_g, center_pix_coord, dp_gauss_rgba.d);
                d_xy_vs = dp_g.d.xy_vs;
                d_inv_cov_vs_opacity = dp_g.d.inv_cov_vs_opacity;
                d_rgb = dp_g.d.rgb;

                }}
            }

            // add back
        #if EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD

          #if SUBGROUP_REDUCE_BEFORE_ATOMIC
            #define subgroupSumF32(temp, val) \
                temp = WaveActiveSum(val);
          #else
            #define subgroupSumF32(temp, val) \
                temp = val;
          #endif

          #if SKIP_ZERO_IN_ATOMIC
            #define ifNonZero(x) if (x != 0.0)
          #else
            #define ifNonZero(x)
          #endif

            float temp_p0, temp_p1;
            float temp_s0, temp_s1, temp_s2, temp_s3;
            float temp_c0, temp_c1, temp_c2;
            subgroupSumF32(temp_p0, d_xy_vs.x);
            subgroupSumF32(temp_p1, d_xy_vs.y);
            subgroupSumF32(temp_s0, d_inv_cov_vs_opacity.x);
            subgroupSumF32(temp_s1, d_inv_cov_vs_opacity.y);
            subgroupSumF32(temp_s2, d_inv_cov_vs_opacity.z);
            subgroupSumF32(temp_s3, d_inv_cov_vs_opacity.w);
            subgroupSumF32(temp_c0, float(d_rgb.x));
            subgroupSumF32(temp_c1, float(d_rgb.y));
            subgroupSumF32(temp_c2, float(d_rgb.z));
          #if SUBGROUP_REDUCE_BEFORE_ATOMIC
            // if (WaveIsFirstLane()) {
            if (thread_rank % SUBGROUP_SIZE == 0) {
          #else
            if (true) {
          #endif
              #if USE_EMULATED_F32_ATOMIC
                #define _ATOMIC_ADD(buffer, address, value) \
                    ifNonZero(value) AtomicAddFloat(buffer, (address)*sizeof(float), value);
              #else
                #define _ATOMIC_ADD(buffer, address, value) \
                    ifNonZero(value) buffer.InterlockedAddF32((address)*sizeof(float), value);
              #endif
                _ATOMIC_ADD(v_xy_vs, 2*g_idx+0, temp_p0);
                _ATOMIC_ADD(v_xy_vs, 2*g_idx+1, temp_p1);
                _ATOMIC_ADD(v_inv_cov_vs_opacity, 4*g_idx+0, temp_s0);
                _ATOMIC_ADD(v_inv_cov_vs_opacity, 4*g_idx+1, temp_s1);
                _ATOMIC_ADD(v_inv_cov_vs_opacity, 4*g_idx+2, temp_s2);
                _ATOMIC_ADD(v_inv_cov_vs_opacity, 4*g_idx+3, temp_s3);
                _ATOMIC_ADD(v_rgb, 3*g_idx+0, temp_c0);
                _ATOMIC_ADD(v_rgb, 3*g_idx+1, temp_c1);
                _ATOMIC_ADD(v_rgb, 3*g_idx+2, temp_c2);
            }

        #endif  // EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD
        }
        splats_left_to_process -= block_size;
    }
}
