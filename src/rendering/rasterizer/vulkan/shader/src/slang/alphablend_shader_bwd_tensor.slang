// Presets: 0 8 0; 0 8 8; 1 16 0;

// 0 or 1
#ifndef USE_SUBGROUP_OPERATIONS
#define USE_SUBGROUP_OPERATIONS 0
#endif

// power-of-2 \in [1, SUBGROUP_SIZE]
#ifndef SPLAT_BATCH_SIZE
#define SPLAT_BATCH_SIZE 8
#endif

// 0: disable group reduce
// power-of-2 \in [1, 256/SPLAT_BATCH_SIZE): reduce to the number
#ifndef GROUP_REDUCE_BEFORE_ATOMIC
#define GROUP_REDUCE_BEFORE_ATOMIC 0
#endif


#define PIXEL_NUM_BATCHES (TILE_SIZE / SPLAT_BATCH_SIZE)
#define PIXEL_BATCH_SIZE (TILE_SIZE / PIXEL_NUM_BATCHES)

groupshared int32_t collected_idx[TILE_SIZE];  // 1 KB
groupshared float2 collected_xy_vs[TILE_SIZE];  // 2 KB
groupshared float4 collected_inv_cov_vs_opacity[TILE_SIZE];  // 4 KB

groupshared float3 collected_rgb[TILE_SIZE];  // 3 KB or 4 KB

#if !USE_SUBGROUP_OPERATIONS
groupshared float splat_embeds[TILE_SIZE][6];  // 6 KB or 8 KB
#endif

groupshared float2 shared_alpha[SPLAT_BATCH_SIZE][TILE_SIZE];  // 2..64 KB for SPLAT_BATCH_SIZE=1..32
groupshared float3 shared_rgb[TILE_SIZE];  // 3 KB or 4 KB

#if GROUP_REDUCE_BEFORE_ATOMIC
groupshared float reduce_splats[PIXEL_NUM_BATCHES][SPLAT_BATCH_SIZE][6];  // 6 KB or 8 KB
#endif


[ForceInline]
void embedS(float2 c, float4 s, float2 tile_center, out float es[6]) {
    c -= tile_center;
    es[0] = float(log(s.w) - 0.5*s.x*c.x*c.x - s.y*c.x*c.y - 0.5*s.z*c.y*c.y);
    es[1] = float(s.x*c.x + s.y*c.y);
    es[2] = float(s.y*c.x + s.z*c.y);
    es[3] = float(-0.5*s.x);
    es[4] = float(-s.y);
    es[5] = float(-0.5*s.z);
}

[ForceInline]
void embedSBwd(float2 c, float4 s, float2 tile_center, in float v[6], out float2 vc, out float4 vs) {
    c -= tile_center;
    vc = float2(
        (-s.x*c.x - s.y*c.y) * v[0] + s.x * v[1] + s.y * v[2],
        (-s.y*c.x - s.z*c.y) * v[0] + s.y * v[1] + s.z * v[2]
    );
    vs = float4(
        -0.5*c.x*c.x * v[0] + c.x * v[1] - 0.5 * v[3],
        -c.x*c.y * v[0] + c.y * v[1] + c.x * v[2] - v[4],
        -0.5*c.y*c.y * v[0] + c.y * v[2] - 0.5 * v[5],
        v[0] / s.w
    );
}

[ForceInline]
void embedP(int pi, float2 tile_center, uint2 tbound0, uint W, uint H, out float ep[6]) {
    float2 p = float2(pi%TILE_WIDTH - TILE_WIDTH/2, pi/TILE_WIDTH - TILE_HEIGHT/2);
    static const float mask = 1.0;
    ep[0] = float(mask * 1.0f);
    ep[1] = float(mask * p.x);
    ep[2] = float(mask * p.y);
    ep[3] = float(mask * p.x*p.x);
    ep[4] = float(mask * p.x*p.y);
    ep[5] = float(mask * p.y*p.y);
}


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
[numthreads(TILE_SIZE, 1, 1)]
void main(
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform Uniforms uniforms
) {
    const uint32_t H = uniforms.image_height;
    const uint32_t W = uniforms.image_width;
    const uint grid_width = uniforms.grid_width;

    uint tile_idx = groupID.x;
    uint2 tbound0 = uint2(tile_idx%grid_width, tile_idx/grid_width) * uint2(TILE_WIDTH, TILE_HEIGHT);
    float2 tile_center = tbound0 + uint2(TILE_WIDTH, TILE_HEIGHT) / 2;

    uint thread_rank = groupThreadID.x % SPLAT_BATCH_SIZE;
    uint thread_base = groupThreadID.x % SUBGROUP_SIZE - thread_rank; 
    uint pixel_batch_rank = groupThreadID.x / SPLAT_BATCH_SIZE;
    uint group_thread_rank = pixel_batch_rank * SPLAT_BATCH_SIZE + thread_rank;

    float4 pixel_state = float4(0.0);
    float4 v_pixel_state = float4(0.0);
    int n_contrib_fwd = 0;
    {
        const uint pi = group_thread_rank;
        uint2 pix_coord = tbound0 + uint2(pi%TILE_WIDTH, pi/TILE_WIDTH);
        bool is_inside = (pix_coord.x < W && pix_coord.y < H);

        if (is_inside) {
            uint32_t pix_id = pix_coord.y * W + pix_coord.x;
            pixel_state = final_pixel_state[pix_id];
            n_contrib_fwd = n_contributors[pix_id];
            v_pixel_state = v_current_pixel_state[pix_id];
        }
    }

    uint32_t tile_idx_start = tile_ranges[tile_idx];
    uint32_t tile_idx_end = tile_ranges[tile_idx+1];
    uint32_t num_splats_in_tile = tile_idx_end - tile_idx_start;

    const int rounds = ((num_splats_in_tile + SPLAT_BATCH_SIZE - 1) / SPLAT_BATCH_SIZE);

    for (int batch_idx = 0; batch_idx < rounds; batch_idx++) {

        int num_splats = min(SPLAT_BATCH_SIZE, num_splats_in_tile - batch_idx * SPLAT_BATCH_SIZE);
        int32_t splat_id_base = num_splats_in_tile - 1 - batch_idx * SPLAT_BATCH_SIZE;

        // load splats
        // thread 0 loads last splat, thread 1 loads second last, ...
        int32_t splat_id = -1;
        Splat_2D_AlphaBlend splat;
        if (batch_idx % PIXEL_NUM_BATCHES == 0) {
            if (batch_idx > 0)
                GroupMemoryBarrierWithGroupSync();
            int32_t splat_tid = splat_id_base - group_thread_rank;
            splat.xy_vs = float2(0);
            splat.inv_cov_vs_opacity = float4(0, 0, 0, 1);
            if (splat_tid >= 0) {
                splat_id = sorted_gauss_idx[tile_idx_start + splat_tid];
                splat = load_splat_alphablend(splat_id, xy_vs, inv_cov_vs_opacity, rgb);
                collected_idx[group_thread_rank] = splat_id;
                collected_xy_vs[group_thread_rank] = splat.xy_vs;
                collected_inv_cov_vs_opacity[group_thread_rank] = splat.inv_cov_vs_opacity;
                collected_rgb[group_thread_rank] = splat.rgb;
            }
          #if !USE_SUBGROUP_OPERATIONS
            float es[6];
            embedS(splat.xy_vs, splat.inv_cov_vs_opacity, tile_center, es);
            [ForceUnroll] for (uint k = 0; k < 6; k++)
                splat_embeds[group_thread_rank][k] = es[k];
          #endif
            GroupMemoryBarrierWithGroupSync();
        }
        int collected_base = (batch_idx % PIXEL_NUM_BATCHES) * SPLAT_BATCH_SIZE;
        splat.xy_vs = float2(0);
        splat.inv_cov_vs_opacity = float4(0, 0, 0, 1);
        // splat.rgb = float3(0);
        if (thread_rank < num_splats) {
            splat_id = collected_base + thread_rank;
            splat.xy_vs = collected_xy_vs[splat_id];
            splat.inv_cov_vs_opacity = collected_inv_cov_vs_opacity[splat_id];
            // splat.rgb = loadCollectedRGB(splat_id);
        }
      #if USE_SUBGROUP_OPERATIONS
        float es[6];
        embedS(splat.xy_vs, splat.inv_cov_vs_opacity, tile_center, es);
      #endif

        {
            const int pi = group_thread_rank;

            float ep[6];
            embedP(pi, tile_center, tbound0, W, H, ep);
            int n_contrib = n_contrib_fwd;

            float3 v_rgb = v_pixel_state.rgb;
            float v_T = v_pixel_state.w;
            float T = pixel_state.w;

            for (int si = 0; si < num_splats; si++) {
                float a = 0.0;
              #if USE_SUBGROUP_OPERATIONS
                [ForceUnroll] for (int k = 0; k < 6; k++)
                    a += ep[k] * WaveShuffle(es[k], thread_base+si);
              #else
                [ForceUnroll] for (int k = 0; k < 6; k++)
                    a += ep[k] * splat_embeds[collected_base + si][k];
              #endif
                a = clipOpacity(exp(a));

                int splat_tid = splat_id_base - si;
                a *= float(splat_tid >= 0 && splat_tid < n_contrib);
                a *= float(a >= ALPHA_THRESHOLD);
                T /= 1.0 - a;
                
                float3 rgb = collected_rgb[collected_base + si];
                // float3 rgb = WaveReadLaneAt(splat.rgb, si);
                
                float aT = a * T;
                float v_a = aT * (dot(v_rgb, rgb) - v_T) * float(a < ALPHA_BLEND_MAX_OPACITY);
                v_T = v_T * (1.0-a) + a * dot(v_rgb, rgb);
                pixel_state = undo_pixel_state(pixel_state, float4(rgb*a, a));
                
                shared_alpha[si][pi] = float2(aT, v_a);
            }
            v_pixel_state.w = v_T;
        }

        // backward
      #if !USE_SUBGROUP_OPERATIONS
        shared_rgb[group_thread_rank] = v_pixel_state.rgb;
      #endif
        GroupMemoryBarrierWithGroupSync();
        {
            float2 v_xy_vs = float2(0.0);
            float4 v_inv_cov_vs_opacity = float4(0.0);
            float3 v_rgb = float3(0.0);

            const int si = thread_rank;
            float v[6] = {0, 0, 0, 0, 0, 0};

            float2 loaded_aw = float2(0);
            float3 loaded_rgb = float3(0);

        #if !USE_SUBGROUP_OPERATIONS
          #if 0
            for (int pi = pixel_batch_rank * PIXEL_BATCH_SIZE, dpi = 0; dpi < PIXEL_BATCH_SIZE; pi++, dpi++) {
            // for (int pi = pixel_batch_rank; pi < TILE_SIZE; pi += PIXEL_NUM_BATCHES) {
                loaded_aw = shared_alpha[si][pi];
                loaded_rgb = shared_rgb[pi];
          #else  // avoid bank conflict
            for (int wi0 = 0; wi0 < PIXEL_BATCH_SIZE; wi0++) {
                int wi = (wi0 + si) % SPLAT_BATCH_SIZE;
                int pi = pixel_batch_rank * PIXEL_BATCH_SIZE + wi;
                loaded_aw = shared_alpha[si][pi];
                loaded_rgb = shared_rgb[pi];
          #endif
        #else
            float3 loaded_rgb_0 = v_pixel_state.rgb;
            for (int wi0 = 0; wi0 < PIXEL_BATCH_SIZE; wi0++) {
                int wi = (wi0 + si) % SPLAT_BATCH_SIZE;
                int pi = pixel_batch_rank * PIXEL_BATCH_SIZE + wi;
                loaded_aw = shared_alpha[si][pi];
                loaded_rgb = WaveShuffle(loaded_rgb_0, thread_base+wi);
        #endif

                float a = loaded_aw.x, w = loaded_aw.y;
                // if (a == 0.0) continue;

                float ep[6];
                embedP(pi, tile_center, tbound0, W, H, ep);
                for (int k = 0; k < 6; k++)
                    v[k] += w * ep[k];
                v_rgb += a * loaded_rgb;
            }

            embedSBwd(
                splat.xy_vs, splat.inv_cov_vs_opacity, tile_center,
                v, v_xy_vs, v_inv_cov_vs_opacity
            );

            // if (thread_rank >= num_splats) continue;
            if (thread_rank >= num_splats) {
                v_xy_vs = float2(0);
                v_inv_cov_vs_opacity = float4(0);
                v_rgb = float3(0);
            }

            if (!(splat_id >= 0))
                continue;
          #if GROUP_REDUCE_BEFORE_ATOMIC > 0
            v = { v_xy_vs.x, v_xy_vs.y, v_inv_cov_vs_opacity.x, v_inv_cov_vs_opacity.y,
                  v_inv_cov_vs_opacity.z, v_inv_cov_vs_opacity.w };
            [ForceUnroll] for (int k = 0; k < 6; k++)
                reduce_splats[pixel_batch_rank][thread_rank][k] = v[k];
            shared_rgb[group_thread_rank] = v_rgb;
            [ForceUnroll]
            for (int s = PIXEL_NUM_BATCHES >> 1; s >= GROUP_REDUCE_BEFORE_ATOMIC; s >>= 1) {
                GroupMemoryBarrierWithGroupSync();
                if (pixel_batch_rank < s) {
                    [ForceUnroll] for (int k = 0; k < 6; k++)
                        v[k] = reduce_splats[pixel_batch_rank][thread_rank][k] +
                            reduce_splats[pixel_batch_rank+s][thread_rank][k];
                    v_rgb = shared_rgb[group_thread_rank] + \
                        shared_rgb[group_thread_rank+s*PIXEL_BATCH_SIZE];
                    if (s > GROUP_REDUCE_BEFORE_ATOMIC) {
                        [ForceUnroll] for (int k = 0; k < 6; k++)
                            reduce_splats[pixel_batch_rank][thread_rank][k] = v[k];
                        shared_rgb[group_thread_rank] = v_rgb;
                    }
                }
            }
            if (pixel_batch_rank < GROUP_REDUCE_BEFORE_ATOMIC) {
                v_xy_vs = float2(v[0], v[1]);
                v_inv_cov_vs_opacity = float4(v[2], v[3], v[4], v[5]);
          #else  // GROUP_REDUCE_BEFORE_ATOMIC
            {
          #endif  // GROUP_REDUCE_BEFORE_ATOMIC
                splat_id = collected_idx[splat_id];

              #if USE_EMULATED_F32_ATOMIC
                #define _ATOMIC_ADD(buffer, address, value) \
                    if (value != 0.0) AtomicAddFloat(buffer, (address)*sizeof(float), value);
              #else
                #define _ATOMIC_ADD(buffer, address, value) \
                    if (value != 0.0) buffer.InterlockedAddF32((address)*sizeof(float), value);
              #endif

                _ATOMIC_ADD(out_v_xy_vs, 2*splat_id+0, v_xy_vs.x);
                _ATOMIC_ADD(out_v_xy_vs, 2*splat_id+1, v_xy_vs.y);
                _ATOMIC_ADD(out_v_inv_cov_vs_opacity, 4*splat_id+0, v_inv_cov_vs_opacity.x);
                _ATOMIC_ADD(out_v_inv_cov_vs_opacity, 4*splat_id+1, v_inv_cov_vs_opacity.y);
                _ATOMIC_ADD(out_v_inv_cov_vs_opacity, 4*splat_id+2, v_inv_cov_vs_opacity.z);
                _ATOMIC_ADD(out_v_inv_cov_vs_opacity, 4*splat_id+3, v_inv_cov_vs_opacity.w);
                _ATOMIC_ADD(out_v_rgb, 3*splat_id+0, v_rgb.x);
                _ATOMIC_ADD(out_v_rgb, 3*splat_id+1, v_rgb.y);
                _ATOMIC_ADD(out_v_rgb, 3*splat_id+2, v_rgb.z);
            }
        }

    }  // for (int i = 0; i < rounds; i++)

}
