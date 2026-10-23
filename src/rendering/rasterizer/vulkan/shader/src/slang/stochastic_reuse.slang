/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "config.slang"
import utils;

// SteadySplats Algorithm 2, in gather form. A thread owns one source pixel's
// deduplicated FIFO. It publishes four outgoing weights; the resolve pass
// gathers them, avoiding floating-point atomics and cross-workgroup races.
static const uint HISTORY_SIZE = 4u;
static const uint EMPTY_ID = 0xfffffffeu;
static const uint BACKGROUND_ID = 0xffffffffu;
static const int2 OFFSETS[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};

layout(binding=0) StructuredBuffer<uint> sample_ids;
layout(binding=1) StructuredBuffer<float> sample_depths;
layout(binding=2) StructuredBuffer<float2> positions;
layout(binding=3) StructuredBuffer<float4> conics;
layout(binding=4) RWStructuredBuffer<uint2> history_ids_depths;
layout(binding=5) RWStructuredBuffer<float4> history_occlusion;
layout(binding=6) RWStructuredBuffer<float4> outgoing_weights;

float fragment_alpha(uint id, float2 pixel, Uniforms uniforms) {
    if (id == BACKGROUND_ID)
        return 1.0f;
    float4 conic = conics[id];
    float alpha = (uniforms.lod_enabled & 4u) != 0u
        ? evaluate_splat_alpha_spark_rad(positions[id], conic.w, conic.xyz, pixel)
        : evaluate_splat_alpha_for_profile(positions[id], conic.w, conic.xyz, pixel,
                                           uniforms.splat_render_profile);
    return alpha >= ALPHA_THRESHOLD ? alpha : 0.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform Uniforms uniforms) {
    uint p = dispatchThreadID.x;
    uint count = uniforms.image_width * uniforms.image_height;
    if (p >= count)
        return;
    uint2 pixel = uint2(p % uniforms.image_width, p / uniforms.image_width);
    float2 origin = full_frame_binning(uniforms)
        ? float2(uniforms.render_origin_x, uniforms.render_origin_y) : float2(0.0f);
    float2 xy = float2(pixel) + origin;
    uint id = sample_ids[p];
    float depth = sample_depths[p];
    float source_alpha = fragment_alpha(id, xy, uniforms);
    float4 target_alpha;
    for (uint d = 0; d < 4; ++d)
        target_alpha[d] = fragment_alpha(id, xy + float2(OFFSETS[d]), uniforms);
    float4 weights = source_alpha > 0.0f ? target_alpha / source_alpha : float4(0.0f);

    bool unique = id != BACKGROUND_ID;
    for (uint k = 0; k < HISTORY_SIZE; ++k) {
        uint address = p * HISTORY_SIZE + k;
        if (uniforms.stochastic_history_valid == 0u) {
            history_ids_depths[address] = uint2(EMPTY_ID, 0u);
            history_occlusion[address] = float4(1.0f);
        }
        uint2 entry = history_ids_depths[address];
        if (entry.x == id)
            unique = false;
        // Empty slots and the sample itself never contribute an occlusion
        // factor. Repeated samples do not refresh or evict the rare entries.
        float stored_depth = asfloat(entry.y);
        if (entry.x != EMPTY_ID && entry.x != id &&
            (stored_depth < depth || (stored_depth == depth && entry.x < id)))
            weights *= history_occlusion[address];
    }
    // With K=4 and alpha capped below one, these products fit in float32.
    outgoing_weights[p] = weights;

    if (unique) {
        for (uint k = HISTORY_SIZE - 1u; k > 0u; --k) {
            history_ids_depths[p * HISTORY_SIZE + k] = history_ids_depths[p * HISTORY_SIZE + k - 1u];
            history_occlusion[p * HISTORY_SIZE + k] = history_occlusion[p * HISTORY_SIZE + k - 1u];
        }
        history_ids_depths[p * HISTORY_SIZE] = uint2(id, asuint(depth));
        history_occlusion[p * HISTORY_SIZE] = (1.0f - target_alpha) / (1.0f - source_alpha);
    }
}
