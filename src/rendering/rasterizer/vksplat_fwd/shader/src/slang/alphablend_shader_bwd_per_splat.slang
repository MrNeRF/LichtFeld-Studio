#undef SUBTILE_SIZE
#define SUBTILE_SIZE 0

// Time: (T/P) * (N+S)/S * (P+S)
// P: larger is better, up to TILE_SIZE
// S: closer to sqrt(N*P) is better
// S should be a multiple of SUBGROUP_SIZE, round up

#define PIXEL_BATCH_SIZE TILE_SIZE

static const uint MAX_SPLAT_BATCH_SIZE = 128;


groupshared float4 pixel_state[PIXEL_BATCH_SIZE];
groupshared float4 v_pixel_state[PIXEL_BATCH_SIZE];
groupshared int n_contrib_fwd[PIXEL_BATCH_SIZE];

layout(binding=0) StructuredBuffer<int32_t> sorted_gauss_idx;
layout(binding=1) StructuredBuffer<int32_t> tile_ranges;
layout(binding=2) StructuredBuffer<float2> xy_vs;
layout(binding=3) StructuredBuffer<float4> inv_cov_vs_opacity;
layout(binding=4) StructuredBuffer<float> rgb;
layout(binding=5) StructuredBuffer<float4> final_pixel_state;
layout(binding=6) StructuredBuffer<int32_t> n_contributors;
layout(binding=7) StructuredBuffer<float4> v_current_pixel_state;

layout(binding=8) RWByteAddressBuffer out_v_xy_vs;
layout(binding=9) RWByteAddressBuffer out_v_inv_cov_vs_opacity;
layout(binding=10) RWByteAddressBuffer out_v_rgb;


[shader("compute")]
[numthreads(MAX_SPLAT_BATCH_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform Uniforms uniforms
) {
    const uint32_t H = uniforms.image_height;
    const uint32_t W = uniforms.image_width;
    const uint grid_width = uniforms.grid_width;

    uint thread_rank = groupThreadID.x;
    uint2 tbound0 = uint2(groupID.x%grid_width, groupID.x/grid_width) * uint2(TILE_WIDTH, TILE_HEIGHT);

    [ForceUnroll]
    for (int i = 0; i < TILE_SIZE; i += MAX_SPLAT_BATCH_SIZE) {
        int pix_rank_batch = i + thread_rank;

        int pix_rank = pix_rank_batch;
        uint2 pix_coord = tbound0 + uint2(pix_rank%TILE_WIDTH, pix_rank/TILE_WIDTH);
        bool is_inside = (pix_coord.x < W && pix_coord.y < H);

        if (is_inside) {
            uint32_t pix_id = pix_coord.y * W + pix_coord.x;
            pixel_state[pix_rank_batch] = final_pixel_state[pix_id];
            v_pixel_state[pix_rank_batch] = v_current_pixel_state[pix_id];
            n_contrib_fwd[pix_rank_batch] = n_contributors[pix_id];
        }
        else {
            n_contrib_fwd[pix_rank_batch] = 0;
        }

    }
    GroupMemoryBarrierWithGroupSync();

    uint tile_idx = groupID.y * grid_width + groupID.x;
    uint32_t tile_idx_start = tile_ranges[tile_idx];
    uint32_t tile_idx_end = tile_ranges[tile_idx+1];
    uint32_t num_splats_in_tile = tile_idx_end - tile_idx_start;

    #ifdef SPLAT_BATCH_SIZE
      #undef SPLAT_BATCH_SIZE
    #endif
    int SPLAT_BATCH_SIZE = int(sqrt(num_splats_in_tile * PIXEL_BATCH_SIZE));
    SPLAT_BATCH_SIZE = ((SPLAT_BATCH_SIZE + SUBGROUP_SIZE) / SUBGROUP_SIZE) * SUBGROUP_SIZE;
    SPLAT_BATCH_SIZE = min(SPLAT_BATCH_SIZE, MAX_SPLAT_BATCH_SIZE);
    bool active = (thread_rank < SPLAT_BATCH_SIZE);

    const int rounds = ((num_splats_in_tile + SPLAT_BATCH_SIZE - 1) / SPLAT_BATCH_SIZE);

    [ForceUnroll]
    for (int pixel_batch_base = 0; pixel_batch_base < TILE_SIZE; pixel_batch_base += PIXEL_BATCH_SIZE) {

    for (int batch_idx = 0; batch_idx < rounds; batch_idx++) {

        // load splat
        // thread 0 loads last splat, thread 1 loads second last, ...
        int32_t splat_id = -1;
        Splat_2D_AlphaBlend splat;
        int32_t progress = batch_idx * SPLAT_BATCH_SIZE + thread_rank;
        int32_t splat_gid = tile_idx_end - progress - 1;
        int32_t splat_tid = splat_gid - tile_idx_start;
        if (active & (splat_tid >= 0)) {
            splat_id = sorted_gauss_idx[splat_gid];
            splat = load_splat_alphablend(splat_id, xy_vs, inv_cov_vs_opacity, rgb);
        }

        // compute subtile culling masks
        #if SUBTILE_SIZE > 0
        uint32_t subtile_overlap_bitmask = 0;
        if (active) subtile_overlap_bitmask = computeTileOverlapBitmask(
            false, splat.xy_vs - tbound0, splat.inv_cov_vs_opacity);
        #endif

        float2 v_xy_vs = float2(0.0);
        float4 v_inv_cov_vs_opacity = float4(0.0);
        float3 v_rgb = float3(0.0);

        float4 current_pixel_state;
        float4 d_current_pixel_state;

        int num_pixels = PIXEL_BATCH_SIZE;
        int num_splats = min(SPLAT_BATCH_SIZE, num_splats_in_tile - batch_idx * SPLAT_BATCH_SIZE);

        // render: s0 last, s1, second last, ...
        // backward: p first sees s0, then s1, then s2...

        // Example with 5 splats, 4 pixels
        // i=0   s:    01234
        //       p: 3210
        // i=1   s:    01234
        //       p:  3210
        // i=2   s:    01234
        //       p:   3210
        // i=3   s:    01234
        //       p:    3210
        // i=4   s:    01234
        //       p:     3210
        // i=5   s:    01234
        //       p:      3210
        // i=6   s:    01234
        //       p:       3210
        // i=7   s:    01234
        //       p:        3210

        for (int i = 0; i < num_splats + num_pixels - 1; ++i, GroupMemoryBarrierWithGroupSync()) {

            int splat_rank = thread_rank;
            int pixel_rank = i - thread_rank;
            if (!active || splat_rank >= num_splats || pixel_rank >= num_pixels || pixel_rank < 0)
                continue;

            int pixel_idx = pixel_batch_base + pixel_rank;
            uint2 pix_coord = tbound0 + uint2(pixel_idx%TILE_WIDTH, pixel_idx/TILE_WIDTH);
            uint pix_rank = pixel_rank;
          #if SUBTILE_SIZE > 0
            uint subgrid_idx;
            decodeThreadToSubtiles(pixel_rank, pix_coord, subgrid_idx, pix_rank);
            int overlap = (subtile_overlap_bitmask >> subgrid_idx) & 1;
            if (overlap == 0)
                continue;
          #endif

            uint32_t n_contrib = n_contrib_fwd[pix_rank];
            // uint32_t n_contrib = (pix_coord.x >= W || pix_coord.y >= H) ? 0 : num_splats_in_tile;
            if (splat_tid < 0 || splat_tid >= n_contrib)
                continue;

            float2 center_pix_coord = pix_coord;
            float4 gauss_rgba = evaluate_splat(splat, center_pix_coord);
            if (gauss_rgba.a <= ALPHA_THRESHOLD)
                continue;
            
            current_pixel_state = pixel_state[pix_rank];
            current_pixel_state = undo_pixel_state(current_pixel_state, gauss_rgba);
            pixel_state[pix_rank] = current_pixel_state;

            d_current_pixel_state = v_pixel_state[pix_rank];
            DifferentialPair<float4> dp_gauss_rgba = diffPair(gauss_rgba);
            DifferentialPair<float4> dp_current_pixel_state = diffPair(current_pixel_state);
            bwd_diff(update_pixel_state)(dp_current_pixel_state, dp_gauss_rgba, d_current_pixel_state);
            v_pixel_state[pix_rank] = dp_current_pixel_state.d;

            DifferentialPair<Splat_2D_AlphaBlend> dp_g = diffPair(splat);
            bwd_diff(evaluate_splat)(dp_g, center_pix_coord, dp_gauss_rgba.d);
            v_xy_vs += dp_g.d.xy_vs;
            v_inv_cov_vs_opacity += dp_g.d.inv_cov_vs_opacity;
            v_rgb += dp_g.d.rgb;
        }

      #if USE_EMULATED_F32_ATOMIC
        #define _ATOMIC_ADD(buffer, address, value) \
            if (value != 0.0) AtomicAddFloat(buffer, (address)*sizeof(float), value);
      #else
        #define _ATOMIC_ADD(buffer, address, value) \
            if (value != 0.0) buffer.InterlockedAddF32((address)*sizeof(float), value);
      #endif
        if (active) {
            _ATOMIC_ADD(out_v_xy_vs, 2*splat_id+0, v_xy_vs.x);
            _ATOMIC_ADD(out_v_xy_vs, 2*splat_id+1, v_xy_vs.y);
            _ATOMIC_ADD(out_v_inv_cov_vs_opacity, 4*splat_id+0, v_inv_cov_vs_opacity.x);
            _ATOMIC_ADD(out_v_inv_cov_vs_opacity, 4*splat_id+1, v_inv_cov_vs_opacity.y);
            _ATOMIC_ADD(out_v_inv_cov_vs_opacity, 4*splat_id+2, v_inv_cov_vs_opacity.z);
            _ATOMIC_ADD(out_v_inv_cov_vs_opacity, 4*splat_id+3, v_inv_cov_vs_opacity.w);
            _ATOMIC_ADD(out_v_rgb, 3*splat_id+0, float(v_rgb.x));
            _ATOMIC_ADD(out_v_rgb, 3*splat_id+1, float(v_rgb.y));
            _ATOMIC_ADD(out_v_rgb, 3*splat_id+2, float(v_rgb.z));
        }

    }  // for (int i = 0; i < rounds; i++)

    }  // pixel batch
}
